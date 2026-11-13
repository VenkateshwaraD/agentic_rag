Classify this client-service inquiry. Return ONLY JSON:
{{"sources": [...], "ops": [...], "client_id": "..."}}

sources: subset of ["agreements","slas","service_teams"] — document
collections to search (fees/statements → agreements; cutoffs/incidents →
slas; team contacts/ownership → service_teams). Use multiple if needed.
ops: subset of ["accounts","trades","tickets"] — systems of record to
query (account details → accounts; payment/transfer status → trades;
case history → tickets).
client_id: the client the question is about, or null if unclear.

Question: {question}
Entitled clients: {clients}
