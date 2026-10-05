SELECT client_type, COUNT(*) AS client_count
FROM clients
GROUP BY client_type;
