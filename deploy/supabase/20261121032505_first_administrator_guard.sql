-- For review only. Apply only to veepwacsdbqnzkblxbiu after approval.
BEGIN;

CREATE OR REPLACE FUNCTION psms_accounts.preserve_administrator()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'pg_catalog'
AS $function$
BEGIN
 IF TG_OP='DELETE' THEN
   IF OLD.role='Admin' THEN RAISE EXCEPTION 'The designated administrator cannot be removed'; END IF;
   RETURN OLD;
 END IF;
 IF TG_OP='INSERT' AND NEW.role='Admin' THEN
   IF session_user <> 'postgres' OR current_user <> 'postgres'
      OR current_setting('transaction_isolation') <> 'read committed'
      OR NEW.username IS DISTINCT FROM 'admin'
      OR NEW.status IS DISTINCT FROM 'Active' THEN
     RAISE EXCEPTION 'Additional administrators are prohibited';
   END IF;
   LOCK TABLE public.users IN SHARE ROW EXCLUSIVE MODE;
   IF EXISTS (SELECT 1 FROM public.users) THEN
     RAISE EXCEPTION 'Additional administrators are prohibited';
   END IF;
 END IF;
 IF TG_OP='UPDATE' AND NEW.role IS DISTINCT FROM OLD.role
    AND (OLD.role='Admin' OR NEW.role='Admin') THEN
   RAISE EXCEPTION 'The designated administrator cannot be changed or replaced';
 END IF;
 IF TG_OP='UPDATE' AND OLD.role='Admin' AND NEW.status IS DISTINCT FROM 'Active' THEN
   RAISE EXCEPTION 'The designated administrator cannot be disabled by account updates';
 END IF;
 RETURN NEW;
END;
$function$;

COMMIT;
