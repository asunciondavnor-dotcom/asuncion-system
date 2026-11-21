-- New-project bootstrap after the empty baseline verifier passes.
BEGIN;

INSERT INTO psms_workbooks.heads (kind, revision)
VALUES ('inventory_source', 0), ('ppmp_source', 0);

INSERT INTO psms_sync.revisions (singleton, values_json)
VALUES (true, '{}'::jsonb);

INSERT INTO psms_signatories.config_state (id, revision)
VALUES (1, 0);

COMMIT;
