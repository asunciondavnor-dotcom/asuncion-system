"""One private PPMP source, item-level category review, existing official workflow."""
import hashlib
import json
import re
import uuid
from collections import defaultdict
from io import BytesIO
import time

from flask import abort, flash, g, make_response, redirect, render_template, request, send_file, session, url_for
import durable_ocr_v0_1_0 as durable
import document_processing as processing
from private_document_uploads import validate, extraction_path

CATEGORIES=('DBM','NON-DBM','LIB')
WORKBOOKS = {'.xlsx', '.xlsm'}
ITEM_COLUMNS = {
    'Item Name': 'Item Name',
    'Description': 'General Description and Objective of the Project to be Procured',
    'Quantity': 'Quantity',
    'Unit': 'Unit',
    'Unit Cost': 'Estimated Unit Cost',
    'Estimated Budget': 'Estimated Budget Allocation (Php)',
    'Project Type': 'Type of the Project to be Procured',
    'Category': None,
}
WORKBOOK_ALIASES = {
    'item': 'item name', 'particulars': 'item name', 'qty': 'quantity',
    'unit of measure': 'unit', 'uom': 'unit', 'unit price': 'unit cost',
    'ppmp category': 'category', 'source category': 'category',
    'estimated budget allocation (php)': 'estimated budget',
    'estimated unit cost': 'unit cost', 'u/m': 'unit',
    'general description and objective of the project to be procured': 'description',
    'type of the project to be procured': 'project type',
}
WORKBOOK_PLANNING_COLUMNS = (
    'Quantity and Size of the Project to be Procured', 'Recommended Mode of Procurement',
    'Pre-Procurement Conference Applicable (Yes/No)', 'Start of Procurement Activity',
    'End of Procurement Activity', 'Expected Delivery/Implementation Period', 'Source of Funds',
    'Attached Supporting Documents', 'Remarks',
)
PHOTO_ITEM = re.compile(r'^\W*(?P<quantity>\d+)\s*(?P<unit>btls?|btis|bts|bils|bls|pcs?|pes|rolls?|sets?|seis|lengths?)\b[.,\s]+(?P<name>\S.*)',re.I)
PHOTO_UNITS = dict.fromkeys(('btl','btls','btis','bts','bils','bls'),'bottles')
PHOTO_UNITS.update(pc='pcs',pes='pcs',roll='rolls',sets='sets',set='sets',seis='sets',length='lengths')


def photographed_item_fields(item,line, alternate):
    """Split the numbered item column without promoting ambiguous quantities."""
    from difflib import SequenceMatcher
    from procurement_item_contract import SPECIFICATION
    first=PHOTO_ITEM.match(line)
    if first is None:return
    fields=item['fields']
    fields['Item Name']=first['name'].strip()
    fields[SPECIFICATION]=line
    unit=PHOTO_UNITS.get(first['unit'].lower(),first['unit'].lower())
    second=PHOTO_ITEM.match(alternate)
    if (unit!='lengths' and second and first['quantity']==second['quantity']
            and unit==PHOTO_UNITS.get(second['unit'].lower(),second['unit'].lower())
            and SequenceMatcher(None,first['name'].casefold(),second['name'].casefold()).ratio()>=.60):
        fields.update(Quantity=first['quantity'],Unit=unit)
    else:
        evidence=item['field_confidence']['_source_evidence']
        evidence['quantity_review']='Confirm the quantity and unit against the original.'
        if alternate and alternate!=line:evidence['alternate_reading']=alternate


def review_item(text, evidence):
    return dict(fields={'Item Name':text,
                        'General Description and Objective of the Project to be Procured':text},
                field_confidence={'_source_evidence':dict(evidence,text=text)})


def inventory_reference_items(document):
    """Recover positioned inventory reference lines without importing balances."""
    from procurement_document_layout import ordered_word_text
    items=[]
    for page in document.get('page_evidence',[]):
        words=page.get('words',[])
        for anchor in words:
            if anchor['text'].casefold()!='on-hand':
                continue
            band=[w for w in words if abs(w['top']-anchor['top'])<=anchor['height']*1.5]
            by_name={w['text'].casefold():w for w in band}
            if not {'item','description','qty','value','excel'}<=by_name.keys():
                continue
            unit=[w['left'] for w in band if w['text'].casefold()=='unit']
            if len(unit)!=2:continue
            left=by_name['item']['left']-3
            unit_x,price_x=sorted(unit)
            row_x=by_name['excel']['left']-3
            if not left<unit_x<anchor['left']<price_x<row_x:continue
            for number in sorted((w for w in words if w['left']>=row_x and w['top']>anchor['top']
                                  and w['text'].isdigit()),key=lambda w:w['top']):
                row=[w for w in words if abs(w['top']-number['top'])<number['height']*.7]
                groups=[[w for w in row if start-3<=w['left']<end-3] for start,end in
                        [(left,unit_x),(unit_x,anchor['left']),(anchor['left'],price_x),(price_x,row_x)]]
                name,unit_text,quantity,value=[ordered_word_text(group).strip() for group in groups]
                if name and unit_text and re.fullmatch(r'[\d,.]+',quantity) and re.fullmatch(r'(?:PHP\s*)?[\d,.]+',value):
                    evidence=dict(page=page.get('page'),row=len(items)+1,source='Inventory reference',
                                  reference_row=number['text'],on_hand=quantity,historical_value=value,source_unit=unit_text)
                    item=review_item(name,evidence)
                    item['fields']['Unit']=unit_text
                    items.append(item)
            break
    return items


def photographed_ppmp_items(path, deadline=None):
    """Second pass on the item column of a photographed, numbered PPMP form."""
    from PIL import Image, ImageOps
    import numpy as np
    from scipy import ndimage
    import pytesseract
    from aoq_flexible import _configure_tesseract
    from offline_ocr_mapper import raster_memory_scope
    def timeout(seconds):
        remaining=seconds if deadline is None else min(seconds,deadline-time.perf_counter())
        if remaining<=0:raise TimeoutError('Item recovery reached the processing deadline.')
        return remaining
    with Image.open(path) as original:
        if original.width<=original.height or original.width*original.height>4_000_000 or getattr(original,'n_frames',1)!=1:return []
        _configure_tesseract()
        with raster_memory_scope(original.width*original.height):
            image=ImageOps.exif_transpose(original).convert('L')
            data=pytesseract.image_to_data(image.crop((0,0,image.width,int(image.height*.49))),
                        config='--oem 3 --psm 11',output_type=pytesseract.Output.DICT,timeout=timeout(18))
            if not re.search(r'project\s+procurement',' '.join(data['text']),re.I):return []
            anchors={}
            for index, value in enumerate(data['text'][:-1]):
                col=data['text'][index+1]
                if value.casefold()=='column' and col in ('2','3','4') and abs(data['top'][index]-data['top'][index+1])<max(5,data['height'][index]):
                    anchors[col]=(data['left'][index],data['top'][index]+data['height'][index],data['height'][index])
            if not {'2','3','4'}<=anchors.keys():return []
            a,b,c=[anchors[key] for key in ('2','3','4')]
            if not a[0]<b[0]<c[0] or max(v[1] for v in anchors.values())-min(v[1] for v in anchors.values())>3*b[2]:return []
            left=max(0,round(b[0]-(b[0]-a[0])*.52))
            right=round(b[0]+(c[0]-b[0])*.76)
            top=b[1]+max(1,b[2]//4)
            bottom=min(image.height,top+round(image.height*.36))
            def prepared_crop(box):
                pixels=np.array(ImageOps.autocontrast(image.crop(box)))
                mask=pixels<np.percentile(pixels,30)
                pixels[ndimage.binary_opening(mask,structure=np.ones((1,19))) |
                       ndimage.binary_opening(mask,structure=np.ones((25,1)))]=255
                return ImageOps.expand(Image.fromarray(pixels).resize((pixels.shape[1]*2,pixels.shape[0]*2)),15,'white')
            if right-left<30 or bottom-top<30:return []
            crop=prepared_crop((left,top,right,bottom))
            lines=pytesseract.image_to_string(crop,config='--oem 3 --psm 6',timeout=timeout(15)).splitlines()
            checks=[]
            try:
                check_crop=prepared_crop((max(0,left-2),top+1,right-5,bottom-10))
                checks=pytesseract.image_to_string(check_crop,config='--oem 3 --psm 6',
                                                     timeout=timeout(13)).splitlines()
            except (RuntimeError,TimeoutError):
                pass
    def item_lines(values):
        return [line.strip(' |[]_~\t') for line in values if re.search(
            r'\b(?:btls?|btis|bts|bils|bls|pcs?|pes|rolls?|sets?|lengths?)\b|^\W*\d+\s*seis\b',line,re.I)]
    lines,checks=item_lines(lines),item_lines(checks)
    candidates=[]
    from difflib import SequenceMatcher
    next_check=0
    for index,line in enumerate(lines):
        item=review_item(line,dict(source='Photographed PPMP item column',page=1,row=index+1,
                                   bbox=[left,top,right,bottom]))
        alternate=''
        primary=PHOTO_ITEM.match(line)
        if primary:
            scores=[]
            for offset,check in enumerate(checks[next_check:next_check+3],next_check):
                reading=PHOTO_ITEM.match(check)
                if reading:
                    score=SequenceMatcher(None,primary['name'].casefold(),reading['name'].casefold()).ratio()
                    if score>=.72:scores.append((score,-offset,check))
            if scores:
                _,negative_offset,alternate=max(scores)
                next_check=1-negative_offset
        photographed_item_fields(item,line,alternate)
        candidates.append(item)
    return candidates


def workbook_items(document):
    """Read explicit item columns, retaining the original worksheet and row."""
    items = []
    for table in document.get('tables', []):
        rows = table['rows']
        for index, row in enumerate(rows):
            headers = [str(value).strip().casefold() for value in row]
            headers = [WORKBOOK_ALIASES.get(header,header) for header in headers]
            if 'item name' in headers and 'quantity' in headers:
                break
        else:
            continue
        columns = {**ITEM_COLUMNS,**{label:label for label in WORKBOOK_PLANNING_COLUMNS}}
        positions = {key: headers.index(key.casefold()) for key in columns
                     if headers.count(key.casefold()) == 1}
        if not {'Item Name', 'Quantity'} <= positions.keys():
            continue
        cell_evidence = defaultdict(list)
        for cell in table.get('cells', []):
            cell_evidence[cell.get('worksheet_row')].append(cell)
        numbers = table.get('row_numbers') or range(1, len(rows)+1)
        for offset, row in enumerate(rows[index + 1:], index + 1):
            values = {key: str(row[pos] if pos < len(row) and row[pos] is not None else '').strip()
                      for key, pos in positions.items()}
            if not values['Item Name'] or not any(values.get(key) for key in ('Quantity', 'Unit', 'Unit Cost')):
                continue
            fields = {label: values[key] for key, label in columns.items() if label and key in values}
            if values['Item Name'].casefold() == 'item name' and values['Quantity'].casefold() == 'quantity':
                continue
            line = numbers[offset]
            evidence = dict(worksheet=table.get('worksheet'), row=line, headers=[str(v) for v in rows[index]],
                            cells=row, text=' | '.join(str(v) for v in row),
                            cell_evidence=cell_evidence[line])
            items.append(dict(fields=fields, field_confidence={'_source_evidence': evidence},
                              native_category=category(values.get('Category'))))
            if len(items) > 200:
                abort(400, description='Review at most 200 items per upload. Split the workbook into smaller files.')
    return items


def category(value):
    text=re.sub(r'[\s\u2010-\u2015]+','-',str(value or '').strip().upper())
    if text=='NONDBM':text='NON-DBM'
    return text if text in CATEGORIES else None


def suggest_category(row):
    """Use explicit row, worksheet or scoped document evidence; never item-name guesses."""
    if row.get('native_category'):
        return row['native_category'], 'explicit workbook category'
    confidence=row.get('field_confidence') or {}
    evidence=confidence.get('_source_evidence') or {}
    candidates=[]
    cells=str(evidence.get('text') or '').split(' | ')
    explicit_columns=False
    for index,label in enumerate(evidence.get('headers') or []):
        if re.sub(r'[^a-z]+',' ',str(label).lower()).strip() in ('category','ppmp category','source category'):
            explicit_columns=True
            value=category(cells[index]) if index<len(cells) else None
            if not value:return None,'source item category is missing or unsupported; manual assignment required'
            candidates.append((value,'item category column'))
    # Explicit item columns take precedence over shared document context.
    if not candidates and not explicit_columns:
        value=category((confidence.get('_document_metadata') or {}).get('category'))
        if value:candidates.append((value,'scoped source category'))
    if not candidates:
        value=category(evidence.get('worksheet'))
        if value:candidates.append((value,'worksheet category'))
    values={value for value,_ in candidates}
    return (next(iter(values)),candidates[0][1]) if len(values)==1 else (None,'manual category assignment required')


def item_identity(row):
    evidence=(row.get('field_confidence') or {}).get('_source_evidence') or {}
    number=str(evidence.get('item_number') or '').strip()
    if number.isdigit():number=str(int(number))
    scope=[str(evidence.get('worksheet') or ''),str(evidence.get('page') or '')]
    anchor=['item',number] if number else ['row',str(evidence.get('row') or ''),str(evidence.get('source') or '')]
    if not number and not evidence.get('row'):
        anchor=['text',str(evidence.get('text') or ''),str(row.get('temp_id') or '')]
    return hashlib.sha256(json.dumps(scope+anchor,ensure_ascii=False).encode()).hexdigest()


def catalog_categories(api,rows):
    """Exact catalog identity only; contradictory matches remain unclassified."""
    names={str((row.get('fields') or {}).get('Item Name') or '').strip().lower()
           for row in rows if suggest_category(row)[0] is None}
    names.discard('')
    if not names:return {}
    records=api['q']('''select id,article,description,unit,sheet_name,raw_data,normalized_data
        from dataset_records where active=1 and
        (lower(btrim(article))=any(?::text[]) or lower(btrim(description))=any(?::text[]))
        order by id limit 501''',(sorted(names),sorted(names)))
    if len(records)>500:return {}
    result={}
    for record in records:
        values=set();explicit=False
        for data in (api['_json_dict'](record.get('raw_data')),api['_json_dict'](record.get('normalized_data'))):
            for key in ('PPMP Category','ppmp_category','Source Category','category','Category'):
                if key in data:
                    explicit=True
                    # None deliberately preserves a missing/unsupported source
                    # category; a worksheet title must not silently replace it.
                    values.add(category(data[key]))
        if not explicit:
            sheet=category(record.get('sheet_name'))
            if sheet:values.add(sheet)
        for name in (record.get('article'),record.get('description')):
            key=str(name or '').strip().lower()
            if key in names:result.setdefault(key,set()).update(values)
    return result


def save_extraction(api,name,ext,content,extracted,structured,*,source_id=None,run_id=None,timings=None,return_category=None):
    """Persist atomically after extraction; repeated source bytes reuse existing drafts."""
    rows=structured.get('rows') or []
    if return_category is not None and return_category not in CATEGORIES:abort(404)
    def destination(sid):
        return url_for('ppmp_type',ptype=return_category,ocr_source=str(sid)) if return_category else url_for('ppmp_intake_review',sid=str(sid))
    if len(rows)>200:abort(400,description='Review at most 200 items per upload.')
    if ext not in WORKBOOKS and not structured.get('direct_recovery') and not (structured.get('model') or {}).get('active'):
        abort(503,description='The semantic mapper is unavailable. No extraction was saved.')
    digest=hashlib.sha256(content).hexdigest()
    if source_id is None:
        source_id,run_id=processing.capture(api,name,ext,content)
        if run_id is None:return redirect(destination(source_id))
    state=dict(text=str(extracted.get('raw_text') or ''),method=extracted.get('extraction_method') or 'native extraction',
        warnings=structured.get('warnings') or [],filename=name,structured=structured,item_count=len(rows),
        extraction_timestamp=api['now'](),intake='automatic-item-routing-v2',
        parser_version=processing.PARSER_VERSION,timings=timings or {})
    catalog=catalog_categories(api,rows)
    summary=dict(total=len(rows),DBM=0,**{'NON-DBM':0},LIB=0,unclassified=0,
                 inserted=0,preserved=0,updated=0,uncertain_regions=len(structured.get('review_candidates') or []))
    g.document_stage='draft_persistence'
    with processing.transaction(api) as (con,uid):
        processing.lock(con,uid,digest)
        processing.current_run(con,uid,source_id,run_id)
        existing=con.execute('''select * from ppmp_ocr_drafts where source_id=? and owner_id=?
            order by id for update''',(str(source_id),uid)).fetchall()
        prior={}
        for draft in existing:
            confidence=api['_json_dict'](draft['confidence_json'])
            key=confidence.get('_extraction_identity') or item_identity(dict(field_confidence=confidence,fields=api['_json_dict'](draft['fields_json'])))
            prior.setdefault(key,[]).append(draft)
        seen=set();identified=[]
        for row_index,row in enumerate(rows):
            evidence=(row.get('field_confidence') or {}).get('_source_evidence') or {}
            key=item_identity(row)
            if key in seen:
                # Repeated item numbers do not erase distinct physical rows.
                key=hashlib.sha256((key+':'+str(evidence.get('row'))+':'+str(row_index)).encode()).hexdigest()
            seen.add(key);identified.append((key,row))
        unmatched_new=seen-set(prior)
        if existing and unmatched_new and (set(prior)-seen or any(len(group)>1 for group in prior.values())):
            # A parser can change row anchors. Never guess which corrected or
            # official draft a new anchor replaces, or silently duplicate it.
            message='Source item identities changed. Existing drafts and corrections were preserved. Reconcile the saved items before reprocessing.'
            response=make_response(render_template('operation_error.html',active='ppmp',title='Source reconciliation required',
                message=message,status=409),409)
            response.headers['Cache-Control']='private, no-store'
            abort(response)
        for key,row in identified:
            fields=dict(row.get('fields') or {})
            confidence=dict(row.get('field_confidence') or {})
            confidence['_intake_approval_required']=True
            confidence['_original_fields']=dict(fields)
            from offline_ocr_mapper import KEY_TO_LABEL
            project_label=KEY_TO_LABEL['project_type']
            if category(fields.get(project_label)):
                fields[project_label]=''
                confidence[project_label]=dict(score=0,level='Unresolved',source='PPMP category is not procurement project type')
            proposed,reason=suggest_category(row)
            evidence=confidence.get('_source_evidence') or {}
            explicit_headers=[i for i,label in enumerate(evidence.get('headers') or [])
                              if re.sub(r'[^a-z]+',' ',str(label).lower()).strip() in ('category','ppmp category','source category')]
            if not proposed and not explicit_headers:
                matches=catalog.get(str(fields.get('Item Name') or '').strip().lower(),set())
                if len(matches)==1 and next(iter(matches)) in CATEGORIES:
                    proposed,reason=next(iter(matches)),'exact existing procurement catalog evidence'
            confidence['_category_suggestion']=dict(category=proposed,evidence=reason)
            confidence['_category_confirmed']=False
            confidence['_category_assignment']=dict(category=None,method='awaiting_approval',at=api['now']())
            confidence['_extraction_identity']=key
            summary['unclassified']+=1
            status='NEEDS_REVIEW'
            matches=prior.get(key,[])
            if matches:
                draft=matches[0]
                previous=api['_json_dict'](draft['confidence_json'])
                if (len(matches)>1 or draft['revision']>1 or draft['reviewed'] or draft['converted']
                        or draft['status'] in ('CANCELLED','CONVERTED_TO_PPMP') or previous.get('_category_confirmed')):
                    summary['preserved']+=1;continue
                confidence['_reextraction_history']=[dict(run_id=run_id,at=api['now'](),
                    before_fields=api['_json_dict'](draft['fields_json']),before_category=draft['ppmp_category'])]
                con.execute('''update ppmp_ocr_drafts set ppmp_category=?,fields_json=?,confidence_json=?,
                    status=?,reviewed=false,updated_at=?,revision=revision+1 where id=?''',
                    ('UNASSIGNED',durable.payload(fields),durable.payload(confidence),status,api['now'](),draft['id']))
                summary['updated']+=1;continue
            did=con.execute(f'''insert into ppmp_ocr_drafts(upload_token,temp_code,ppmp_category,fields_json,confidence_json,
                average_confidence,source_filename,raw_ocr_text,extraction_method,status,reviewed,created_by,created_at,updated_at,owner_id,source_id)
                values(?,?,?,?,?,?,?,?,?,'{status}',false,?,?,?,?,?) returning id''',
                (str(source_id),'OCR-'+str(uuid.uuid4()),'UNASSIGNED',durable.payload(fields),durable.payload(confidence),
                 float(row.get('average_confidence') or 0),name,state['text'],state['method'],session['username'],api['now'](),api['now'](),uid,str(source_id))).fetchone()['id']
            con.execute('update ppmp_ocr_drafts set temp_code=? where id=?',(f'OCR-TEMP-{did:06d}',did))
            summary['inserted']+=1
        result_status=processing.complete(con,uid,run_id,state,summary)
        durable.audit(api,con,'Saved Extraction Drafts for Review','PPMP',source_id)
    if rows:
        flash(f"{summary['inserted']} draft(s) saved; {summary['updated']} refreshed; {summary['preserved']} preserved. Official review and approval remain separate.",'success')
        if result_status=='INCOMPLETE':flash('Extraction is incomplete. Review the retained uncertain source regions.','info')
    else:flash('Document saved. Extraction incomplete: no reliable procurement items were identified. Reprocess or review the original source.','info')
    return redirect(destination(source_id))


def upload(api):
    g.document_stage='validation'
    name,ext,content=validate(request.files.get('ocr_file'))
    return capture_upload(api,name,ext,content)


def capture_upload(api,name,ext,content):
    from nlp_runtime_selection import require_available
    if ext not in WORKBOOKS:
        require_available(api['app'].config)
    from render_model_preparation import ready_for_processing
    with ready_for_processing(None if ext in WORKBOOKS else 'semantic',api['app'].config):
        sid,run_id=processing.capture(api,name,ext,content)
        if run_id is None:
            flash('This document is already saved. Existing corrections were preserved. Use Reprocess Extraction when needed.','info')
            return redirect(url_for('ppmp_intake_review',sid=sid))
        return extract_saved(api,sid,run_id,name,ext,content)


def extract_saved(api,sid,run_id,name,ext,content):
    try:
        deadline=time.perf_counter()+100
        with extraction_path(content,ext) as path:
            g.document_stage='extraction';started=time.perf_counter()
            recovered=[]
            if ext in {'.jpg','.jpeg','.png'}:
                try:recovered=photographed_ppmp_items(path,deadline)
                except (RuntimeError,TimeoutError):pass
            if recovered:
                extracted=dict(raw_text='\n'.join(item['fields']['Item Name'] for item in recovered),
                               extraction_method='OCR of numbered PPMP item column')
                structured=dict(rows=recovered,direct_recovery=True,warnings=[
                    'Compare this OCR text with the original and verify each item, quantity and budget before approval.'])
                return save_extraction(api,name,ext,content,extracted,structured,source_id=sid,run_id=run_id,
                    timings=dict(extraction_seconds=time.perf_counter()-started,mapping_seconds=0))
            extracted=api['extract_aoq_document'](path);native_seconds=time.perf_counter()-started
            g.document_stage='mapping';started=time.perf_counter()
            native=workbook_items(extracted) if ext in WORKBOOKS else None
            structured=(dict(rows=native,warnings=extracted.get('warnings',[]),parser='explicit-excel-columns')
                        if native is not None else api['map_ppmp_ocr_document'](extracted))
            if not structured.get('rows') and (structured.get('model') or {}).get('active'):
                rows = inventory_reference_items(extracted) if ext=='.pdf' else []
                if rows:
                    structured['rows']=rows
                    structured.setdefault('warnings',[]).append(
                        'Compare these recovered rows with the original. Enter verified procurement quantities and budgets before approval.')
            mapping_seconds=time.perf_counter()-started
        return save_extraction(api,name,ext,content,extracted,structured,source_id=sid,run_id=run_id,
                               timings=dict(extraction_seconds=native_seconds,mapping_seconds=mapping_seconds))
    except Exception:
        processing.failed(api,sid,run_id)
        raise


def reprocess(api,sid):
    original=api['q']("select extension from psms_documents.sources where id=? and owner_id=? and kind='ppmp'",
                      (str(sid),durable._uid()),one=True)
    if not original:abort(404)
    if original['extension'] not in WORKBOOKS:
        from nlp_runtime_selection import require_available
        require_available(api['app'].config)
    from render_model_preparation import ready_for_processing
    with ready_for_processing(None if original['extension'] in WORKBOOKS else 'semantic',api['app'].config):
        source,run_id=processing.begin_existing(api,str(sid))
        if run_id is None:
            flash('This parser already completed extraction. Existing drafts and corrections were preserved.','info')
            return redirect(url_for('ppmp_intake_review',sid=str(sid)))
        return extract_saved(api,str(sid),run_id,source['filename'],source['extension'],bytes(source['content']))


def workspace_sources(api):
    return api['q']("""select s.id,s.filename from psms_documents.sources s
        where s.kind='ppmp' and (s.owner_id=? or (? and exists (
            select 1 from ppmp_ocr_drafts d where d.source_id=s.id and d.status='REVIEWED' and not d.converted
            and case when d.confidence_json IS JSON OBJECT then
                d.confidence_json::jsonb->'_intake_verified' is not null
                and d.confidence_json::jsonb->'_item_approval' is null else false end)))
        order by s.created_at desc,s.id desc limit 100""",(durable._uid(),api['is_admin']()))


def workspace_source(api,sid):
    if not isinstance(sid,str) or len(sid)!=36:
        return None
    try:
        identifier = str(uuid.UUID(sid))
    except ValueError:
        return None
    return api['q']("select id from psms_documents.sources where id=? and (owner_id=? or ?) and kind='ppmp'",
                    (identifier,durable._uid(),api['is_admin']()),one=True)


def review(api,sid):
    sid=str(sid)
    admin=api['is_admin']()
    source=api['q']("select id,filename,extension,created_at,owner_id,extraction from psms_documents.sources where id=? and (owner_id=? or ?) and kind='ppmp'",(sid,durable._uid(),admin),one=True)
    if not source:abort(404)
    page=durable.revision(request.args.get('page','1'))
    if not 1<=page<=10000:abort(400)
    if request.method=='POST':
        action=request.form.get('action','verify')
        if action=='approve':
            if not admin:abort(403)
            if request.form.get('confirm_approval')!='yes':abort(400)
            destination=category(request.form.get('category'))
            if len(request.form.getlist('category'))!=1 or not destination:
                abort(400,description='Choose DBM, NON-DBM or LIB before approving this item.')
            with durable.transaction(api) as (con,uid):
                draft=con.execute('select * from ppmp_ocr_drafts where id=? and source_id=? for update',
                                  (durable.revision(request.form.get('draft_id')),sid)).fetchone()
                if not draft:abort(404)
                confidence=api['_json_dict'](draft['confidence_json'])
                if (draft['revision']!=durable.revision(request.form.get('revision')) or draft['converted']
                        or draft['status']!='REVIEWED' or not draft['reviewed']
                        or not confidence.get('_intake_verified') or confidence.get('_item_approval')):
                    abort(409,description='Reopen the latest verified item before approving.')
                fields=api['_json_dict'](draft['fields_json'])
                try:api['validated_ppmp_item_from_review'](fields,destination)
                except ValueError as error:abort(422,description=str(error))
                stamp=api['now']()
                confidence['_item_approval']=dict(by=uid,at=stamp,category=destination)
                confidence['_category_confirmed']=True
                confidence['_category_assignment']=dict(category=destination,method='approval',by=uid,at=stamp)
                con.execute('update ppmp_ocr_drafts set ppmp_category=?,confidence_json=?,updated_at=?,revision=revision+1 where id=?',
                            (destination,durable.payload(confidence),stamp,draft['id']))
                api['log']('Approved Uploaded Item','PPMP',str(draft['id']),'Assigned PPMP category '+destination,connection=con)
            flash(f'Official item approved and assigned to {destination}.','success')
            return redirect(url_for('ppmp_intake_review',sid=sid,page=page))
        if source['owner_id']!=durable._uid():abort(403)
        if action not in ('verify','add'):abort(400)
        if request.form.get('confirm_review',request.form.get('confirm_category'))!='yes':
            abort(400,description='Confirm that the item details match the original.')
        labels=api['ppmp_review_labels']()
        if any('field_'+str(i) not in request.form for i in range(len(labels))):abort(400)
        fields={label:request.form['field_'+str(i)] for i,label in enumerate(labels)}
        if any(len(value)>5000 for value in fields.values()):abort(400)
        fields=api['normalize_reviewed_ocr_fields'](fields)
        try:api['validated_ppmp_item_from_review'](fields,'UNASSIGNED')
        except ValueError as error:abort(422,description=str(error))
        with durable.transaction(api) as (con,uid):
            if action=='add':
                key=hashlib.sha256(durable.payload(fields).encode()).hexdigest()
                con.execute('select pg_advisory_xact_lock(hashtextextended(?,0))',('manual-item:'+sid,))
                if con.execute("""select id from ppmp_ocr_drafts where source_id=? and owner_id=? and
                    case when confidence_json IS JSON OBJECT then confidence_json::jsonb->>'_manual_identity'=? else false end""",
                               (sid,uid,key)).fetchone():
                    abort(409,description='This item is already saved for review.')
                confidence=dict(_original_fields=fields,_manual_identity=key,_intake_approval_required=True,
                                _source_evidence=dict(text='Entered by personnel from the saved original.'))
                did=con.execute('''insert into ppmp_ocr_drafts(upload_token,temp_code,ppmp_category,fields_json,confidence_json,
                    source_filename,extraction_method,status,reviewed,created_by,created_at,updated_at,owner_id,source_id)
                    values(?,?,?,?,?,?,?,'NEEDS_REVIEW',false,?,?,?,?,?) returning id''',
                    (sid,'ITEM-'+uuid.uuid4().hex,'UNASSIGNED',durable.payload(fields),durable.payload(confidence),
                     source['filename'],'Personnel entry',session['username'],api['now'](),api['now'](),uid,sid)).fetchone()['id']
                durable.audit(api,con,'Added Item From Saved Original','PPMP',did)
                return redirect(url_for('ppmp_intake_review',sid=sid,page=page))
            did=durable.revision(request.form.get('draft_id'))
            expected=durable.revision(request.form.get('revision'))
            row=con.execute('select * from ppmp_ocr_drafts where id=? and source_id=? and owner_id=? for update',(did,sid,uid)).fetchone()
            if not row:abort(404)
            if row['revision']!=expected or row['converted'] or row['status'] in ('CANCELLED','CONVERTED_TO_PPMP'):
                abort(409,description='This item changed or became official. Reopen its saved review; your submission was not applied.')
            confidence=api['_json_dict'](row['confidence_json'])
            if confidence.get('_item_approval'):abort(409,description='This official item is already approved.')
            original=confidence.get('_original_fields') or api['_json_dict'](row['fields_json'])
            from procurement_item_contract import decimal_number
            if (decimal_number(fields.get('Quantity'))!=decimal_number(original.get('Quantity')) and
                    request.form.get('confirm_quantity_change')!='yes'):
                abort(400,description='Confirm the intentional quantity difference from the extracted source.')
            confidence['_category_confirmed']=False
            confidence.setdefault('_review_history',[]).append(dict(
                revision=row['revision'],
                by=uid,at=api['now'](),before_fields=api['_json_dict'](row['fields_json']),after_fields=dict(fields)))
            confidence.update(_intake_approval_required=True,_intake_verified=True)
            con.execute("update ppmp_ocr_drafts set ppmp_category='UNASSIGNED',fields_json=?,confidence_json=?,status='REVIEWED',reviewed=true,reviewed_by=?,reviewed_at=?,updated_at=?,revision=revision+1 where id=?",
                        (durable.payload(fields),durable.payload(confidence),session['username'],api['now'](),api['now'](),did))
            durable.audit(api,con,'Verified Uploaded Item','PPMP',did)
        flash('Item verified and saved for authorized approval.','success')
        return redirect(url_for('ppmp_intake_review',sid=sid,page=page))
    extraction,run=processing.latest(api,sid,source['extraction'])
    source=dict(source,extraction=extraction)
    selected=request.args.get('category','ALL')
    selected_sheet=request.args.get('sheet','').strip()
    search=request.args.get('q','').strip()
    sort=request.args.get('sort','source')
    if (selected not in (*CATEGORIES,'ALL','UNASSIGNED') or len(search)>160 or len(selected_sheet)>160
        or any(len(request.args.getlist(key))>1 for key in ('sheet','category','q','sort','page'))):abort(400)
    sorts={'source':'id','name':"lower(fields_json::jsonb->>'Item Name'),id",'name_desc':"lower(fields_json::jsonb->>'Item Name') desc,id",'status':'status,id','newest':'id desc'}
    if sort not in sorts:abort(400)
    source_where='source_id=? and owner_id=?';source_params=[sid,source['owner_id']]
    # A historical malformed/absent evidence payload has no known worksheet;
    # it must not prevent opening All worksheets or be rewritten here. PG17.
    sheet_expression="(case when confidence_json IS JSON OBJECT then confidence_json::jsonb #>> '{_source_evidence,worksheet}' end)"
    sheets=api['q']('select distinct '+sheet_expression+' as worksheet from ppmp_ocr_drafts where '+source_where+
        ' and '+sheet_expression+' is not null order by worksheet',tuple(source_params))
    sheets=[row['worksheet'] for row in sheets if row['worksheet']]
    if selected_sheet and selected_sheet not in sheets:abort(400,description='Choose a saved source worksheet.')
    if selected_sheet:
        source_where+=' and '+sheet_expression+'=?';source_params.append(selected_sheet)
    where=source_where;params=list(source_params)
    assigned="(converted or case when confidence_json IS JSON OBJECT then confidence_json::jsonb->'_item_approval' is not null else false end)"
    if selected=='UNASSIGNED':where+=' and not '+assigned
    elif selected in CATEGORIES:
        where+=' and ppmp_category=? and '+assigned;params.append(selected)
    if search:
        where+=" and strpos(lower(concat_ws(' ','ITEM-'||lpad(id::text,greatest(6,length(id::text)),'0'),temp_code,fields_json)),lower(?))>0";params.append(search)
    summary=api['q'](f'''select count(*) total,
        count(*) filter(where ppmp_category='DBM' and {assigned}) dbm,
        count(*) filter(where ppmp_category='NON-DBM' and {assigned}) non_dbm,
        count(*) filter(where ppmp_category='LIB' and {assigned}) lib,
        count(*) filter(where not {assigned}) unclassified,
        count(*) filter(where converted) official
        from ppmp_ocr_drafts where '''+source_where,tuple(source_params),one=True)
    drafts=api['q']('select * from ppmp_ocr_drafts where '+where+' order by '+sorts[sort]+' limit 21 offset ?',(*params,(page-1)*20))
    has_next=len(drafts)>20;drafts=drafts[:20]
    items=[]
    for row in drafts:
        value=dict(row)
        value['fields']=api['_json_dict'](row['fields_json'])
        value['confidence']=api['_json_dict'](row['confidence_json'])
        value['suggestion']=value['confidence'].get('_category_suggestion') or dict(category=category(row['ppmp_category']),evidence='existing saved category')
        value['source_number']=(value['confidence'].get('_source_evidence') or {}).get('item_number','')
        value['item_reference']=f"ITEM-{row['id']:06d}"
        items.append(value)
    def workspace_url(**changes):
        values={key:request.args[key] for key in request.args if key in ('q','sort','category','page','sheet') or key.startswith('h_')}
        values.update(changes)
        return url_for('ppmp_intake_review',sid=sid,**values)
    if request.method=='GET' and not request.headers.get('X-PSMS-Fragment') and session.get('ppmp_intake_source_id')!=sid:
        session['ppmp_intake_source_id']=sid
    return render_template('ppmp_intake.html',active='ppmp',title='Upload Items',source=source,
        uploads=workspace_sources(api),
        items=items,labels=api['ppmp_review_labels'](),categories=CATEGORIES,page=page,has_next=has_next,
        summary=summary,run=run,selected=selected,selected_sheet=selected_sheet,sheets=sheets,search=search,sort=sort,workspace_url=workspace_url)


def source_file(api,sid):
    row=api['q']("select content,extension from psms_documents.sources where id=? and (owner_id=? or ?) and kind='ppmp'",
                 (str(sid),durable._uid(),api['is_admin']()),one=True)
    if not row:abort(404)
    response=send_file(BytesIO(bytes(row['content'])),as_attachment=True,download_name='source-document'+row['extension'])
    response.headers.update({'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff','Content-Security-Policy':"sandbox; default-src 'none'"})
    return response
