-- Reviewed activation for the new Asuncion database veepwacsdbqnzkblxbiu only.
BEGIN;

INSERT INTO psms_documents.procurement_policy (singleton, policy)
VALUES (true, '{"enabled":true,"po_clock":"ASIA_MANILA","link_authority":"ADMIN_DOCUMENTED"}'::jsonb);

COMMIT;
