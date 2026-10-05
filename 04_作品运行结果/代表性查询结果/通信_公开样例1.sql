SELECT customer_id, customer_name, gender,
       json_extract_string(address_json, '$.province') AS province,
       json_extract_string(address_json, '$.city') AS city
FROM customers
WHERE json_extract_string(address_json, '$.province') = '北京市'
ORDER BY customer_id;
