-- Run after the baseline and before any operational bootstrap.
BEGIN READ ONLY;
SET LOCAL search_path = pg_catalog, public;

DO $verify$
DECLARE
    app_schemas constant text[] := ARRAY[
        'public', 'psms_accounts', 'psms_aoq_api', 'psms_documents',
        'psms_revised_release', 'psms_signatories', 'psms_sync', 'psms_workbooks'
    ];
    table_record record;
    row_present boolean;
    api_role text;
BEGIN
    IF (SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
          WHERE n.nspname = ANY(app_schemas) AND c.relkind = 'r') <> 42 THEN
        RAISE EXCEPTION 'Expected 42 application tables';
    END IF;
    IF to_regclass('psms_documents.procurement_migrations') IS NOT NULL THEN
        RAISE EXCEPTION 'Unexpected legacy deployment ledger';
    END IF;

    FOR table_record IN
        SELECT c.oid, n.nspname, c.relname, c.relrowsecurity
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = ANY(app_schemas) AND c.relkind = 'r'
    LOOP
        IF NOT table_record.relrowsecurity THEN
            RAISE EXCEPTION 'Missing RLS on %.%', table_record.nspname, table_record.relname;
        END IF;
        EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I.%I)', table_record.nspname, table_record.relname)
            INTO row_present;
        IF row_present THEN
            RAISE EXCEPTION 'Baseline must be empty at %.%', table_record.nspname, table_record.relname;
        END IF;
        FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
            IF has_table_privilege(api_role, table_record.oid, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE')
                OR has_any_column_privilege(api_role, table_record.oid, 'SELECT,INSERT,UPDATE') THEN
                RAISE EXCEPTION 'Unexpected API role grant on %.%', table_record.nspname, table_record.relname;
            END IF;
        END LOOP;
    END LOOP;

    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
        IF EXISTS (
            SELECT FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = ANY(app_schemas)
              AND CASE WHEN c.relkind = 'S'
                  THEN has_sequence_privilege(api_role, c.oid, 'USAGE,SELECT,UPDATE')
                  ELSE false END
        ) OR EXISTS (
            SELECT FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = ANY(app_schemas)
              AND (n.nspname <> 'public' OR p.proname IN (
                  'psms_auth_session_permissions_ready', 'psms_revoke_auth_sessions'
              ))
              AND has_function_privilege(api_role, p.oid, 'EXECUTE')
        ) THEN
            RAISE EXCEPTION 'Unexpected application sequence or function grant to %', api_role;
        END IF;
    END LOOP;

    IF EXISTS (
        SELECT FROM pg_roles WHERE rolname IN (
            'psms_vercel_backend', 'psms_account_manager_owner', 'psms_recovery_owner',
            'psms_aoq_function_owner', 'psms_aoq_approval_owner'
        ) AND (rolcanlogin OR rolsuper OR rolbypassrls OR rolcreatedb OR rolcreaterole OR rolreplication)
    ) THEN
        RAISE EXCEPTION 'Unexpected privilege on a new application role';
    END IF;

    IF has_any_column_privilege('psms_vercel_backend', 'public.audit_trail', 'UPDATE')
        OR has_table_privilege('psms_vercel_backend', 'public.audit_trail', 'DELETE,TRUNCATE')
        OR has_any_column_privilege('psms_vercel_backend', 'public.po', 'INSERT,UPDATE')
        OR has_table_privilege('psms_vercel_backend', 'public.po', 'DELETE,TRUNCATE')
        OR has_schema_privilege('psms_vercel_backend', 'public', 'CREATE')
        OR has_table_privilege('psms_vercel_backend', 'psms_revised_release.deployments', 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE')
        OR has_any_column_privilege('psms_vercel_backend', 'psms_revised_release.deployments', 'SELECT,INSERT,UPDATE')
        OR EXISTS (
            SELECT FROM pg_proc
            WHERE pronamespace = 'psms_revised_release'::regnamespace
              AND has_function_privilege('psms_vercel_backend', oid, 'EXECUTE')
        )
        OR pg_has_role('psms_vercel_backend', 'psms_aoq_approval_owner', 'MEMBER')
        OR pg_has_role('psms_vercel_backend', 'psms_aoq_function_owner', 'MEMBER')
        OR pg_has_role('psms_vercel_backend', 'psms_account_manager_owner', 'MEMBER')
        OR pg_has_role('psms_vercel_backend', 'psms_recovery_owner', 'MEMBER') THEN
        RAISE EXCEPTION 'The backend has privileges outside its audit or approval boundaries';
    END IF;

    IF NOT EXISTS (
        SELECT FROM pg_constraint
        WHERE conrelid = 'psms_revised_release.deployments'::regclass
          AND conname = 'deployments_disabled_check'
          AND convalidated
          AND pg_get_constraintdef(oid) = 'CHECK ((NOT enabled))'
    ) THEN
        RAISE EXCEPTION 'The scoped deployment registry must reject enabled rows';
    END IF;
END
$verify$;

SET LOCAL ROLE psms_aoq_approval_owner;
DO $verify$
BEGIN
    PERFORM psms_revised_release.policy_v1();
    RAISE EXCEPTION 'The retired scoped-release policy unexpectedly returned';
EXCEPTION WHEN SQLSTATE 'PA007' THEN
    NULL;
END
$verify$;
RESET ROLE;

SET LOCAL ROLE psms_vercel_backend;
DO $verify$
BEGIN
    IF public.psms_auth_session_permissions_ready() IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'The backend session contract did not pass';
    END IF;
END
$verify$;
RESET ROLE;

SELECT 'empty baseline security checks passed' AS result;
ROLLBACK;
