/* Native submissions retain session validation, CSRF and confirmation. */
(() => {
  'use strict';
  document.addEventListener('submit',event=>{
    const form=event.target;
    if(event.defaultPrevented||!form.matches?.('form[data-document-processing]'))return;
    const notice=form.querySelector('[data-processing-message]');
    if(notice){notice.hidden=false;notice.dataset.category='info';notice.textContent='Processing the document request. Wait for the saved extraction result before continuing.';}
  });
  document.addEventListener('click',event=>{
    const button=event.target.closest('[data-item-dialog]');
    if(!button)return;
    const dialog=document.getElementById(button.dataset.itemDialog);
    if(dialog && !dialog.open)dialog.showModal();
  });
})();
