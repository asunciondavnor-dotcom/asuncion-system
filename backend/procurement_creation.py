"""Parent-serialized child creation. Call only through authenticated Flask routes.

Number allocation, child insertion and action audit share one transaction. The
parent lock is acquired before a separate child-existence statement, so a
request waiting for another creator observes its committed child. Inference
runs outside the transaction, followed by locked source revalidation.
"""
from contextlib import contextmanager
import json
import re
import unicodedata
from flask import abort, flash, redirect, url_for

RELATIONSHIPS = {('app_plan','pr','app_id'), ('aoq','po','aoq_id'),
                 ('po','nlp','po_id'), ('po','iar','po_id')}


@contextmanager
def locked_parent(api, table, identifier, child, foreign_key):
    if (table, child, foreign_key) not in RELATIONSHIPS:
        raise ValueError('Unsupported procurement relationship.')
    if not re.fullmatch(r'[1-9][0-9]{0,18}', str(identifier or '')) or int(identifier) > 9223372036854775807:
        abort(400, description='Select a valid source document.')
    with api['durable_ocr'].transaction(api, read_committed=True) as (con, uid):
        if table=='aoq':
            parent = con.execute('select * from psms_aoq_api.lock_aoq_for_workflow_v1(?::bigint,?::text)',
                                 (identifier,'po')).fetchone()
        elif table=='po':
            import revised_procurement_v1 as revised_procurement
            if revised_procurement.enabled(api):
                try:
                    receiving_lock=('lock_po_for_workflow_v4' if child=='iar' and
                        revised_procurement.tolerance_contract_ready(api) else 'lock_po_for_workflow_v2')
                    parent=con.execute('select * from '+revised_procurement.api_schema()+'.'+receiving_lock+'(?::bigint,?::text)',(identifier,child)).fetchone()
                except Exception as error:
                    state=getattr(error,'sqlstate',None)
                    if state in ('PA002','PA003','PA004'):
                        abort({'PA002':409,'PA003':403,'PA004':404}[state],description='Receiving requires an authorized approved PO and its reviewed verification.')
                    raise
            else:
                parent=con.execute('select * from po where id=? for update',(identifier,)).fetchone()
        else:
            parent = con.execute(f'select * from {table} where id=? for update', (identifier,)).fetchone()
        existing = (con.execute(f'select id,no from {child} where {foreign_key}=? order by id limit 1',
                               (identifier,)).fetchone() if parent else None)
        yield con, parent, existing


def create_pr(api, identifier, purpose=None):
    import revised_procurement_v1 as revised_procurement
    if revised_procurement.enabled(api):
        return revised_procurement.create_pr(api,identifier,purpose)
    purpose = '' if purpose is None else purpose
    if (not isinstance(purpose, str) or len(purpose) > 2000
            or any(unicodedata.category(char) == 'Cc' and char not in '\n\r\t' for char in purpose)):
        flash('Enter a purpose of at most 2,000 characters without unsupported control characters.')
        return redirect(url_for('pr'))
    purpose = purpose.strip()
    with locked_parent(api, 'app_plan', identifier, 'pr', 'app_id') as (con, row, existing):
        if existing or not row or row['status'] != 'Approved':
            flash('Select an approved APP without an existing Purchase Request. No duplicate was created.')
            return redirect(url_for('pr'))
        source = con.execute('select no,items from ppmp where id=? for share', (row['ppmp_id'],)).fetchone()
        if not source:abort(409, description='The source PPMP is unavailable.')
        items = api['parse_items'](row.get('items') or source['items'])
        errors = api['planning_item_errors'](items) if items else ['No approved APP items were found.']
        if errors:
            flash('Cannot create Purchase Request: '+' '.join(errors))
            return redirect(url_for('pr'))
        no = api['next_document_no']('PR','PR','pr',year=row.get('year') or api['current_year'](),connection=con)
        pr_items = api['assign_item_identifiers'](items,'_PR Item ID',no)
        total = api['items_total'](items)
        if total is None or total <= 0:total = api['parse_currency_decimal'](row['total'])
        if total is None or total <= 0:abort(422, description='The approved APP total is invalid.')
        con.execute('''insert into pr(no,app_id,ppmp_id,office,purpose,status,created_at,items,total)
            values(?,?,?,?,?,'For Review',?,?,?)''',
            (no,row['id'],row['ppmp_id'],row['office'],purpose or api['source_pr_purpose'](items),api['now'](),
             json.dumps(pr_items,default=str),total))
        api['log']('Created PR','PR',no,f'Created from approved APP {row["no"]}; source PPMP {source["no"]}; {len(items)} item(s)',connection=con)
    flash(f'Purchase Request {no} created from approved APP {row["no"]}.')
    return redirect(url_for('pr'))


def create_po(api,identifier):
    import revised_procurement_v1 as revised_procurement
    if revised_procurement.enabled(api):
        return revised_procurement.create_po(api,identifier)
    abort(409, description='PO creation requires the authorized revised workflow. The legacy creation path is retired.')


def create_nlp(api,identifier):
    abort(409, description='Select an Approved PR and PO For Review in the authorized revised workflow. Legacy post-approval NLP creation is retired.')


def create_iar(api, identifier):
    import revised_procurement_v1 as revised_procurement

    if not revised_procurement.enabled(api):
        abort(409, description='IAR creation requires the authorized procurement workflow.')
    po_id = revised_procurement.identifier(identifier)
    try:
        with api['durable_ocr'].transaction(api) as (con, _):
            result = con.execute('select * from psms_workflow.create_iar(?)', (po_id,)).fetchone()
    except Exception as error:
        return api['workflow_error'](error, 'iar')
    flash('IAR created for inspection.' if result['changed'] else 'An IAR already exists for this PO.')
    return redirect(url_for('iar_review', id=result['id']))
