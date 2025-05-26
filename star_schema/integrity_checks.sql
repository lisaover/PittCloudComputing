/*
Rows counts match between EMR and fact tables
*/
/* emr and fact_patient */
SELECT patient_id, provider_id, COUNT(lab_id) AS lab_count
FROM emr.visit v
LEFT JOIN emr.visit_lab l
ON v.visit_id = l.visit_id
WHERE EXTRACT(YEAR FROM visit_date) = 2025 AND EXTRACT(MONTH FROM visit_date) = 3 AND patient_id=3 AND provider_id=39
GROUP BY patient_id, provider_id
ORDER BY patient_id, provider_id;

SELECT patient_id, provider_id, lab_count
FROM lao39.fact_patient f
LEFT JOIN lao39.dim_time d
ON d.date_id = f.date_id
WHERE year=2025 AND month='March' AND patient_id=3 AND provider_id=39
ORDER BY patient_id, provider_id;

/* emr and fact_lab */
SELECT provider_id, lab_id, COUNT(patient_id) AS patient_count
FROM emr.visit v
LEFT JOIN emr.visit_lab l
ON v.visit_id = l.visit_id
WHERE EXTRACT(YEAR FROM visit_date) = 2025 AND EXTRACT(MONTH FROM visit_date) = 1 AND lab_id IS NOT NULL
GROUP BY provider_id, lab_id
ORDER BY provider_id, lab_id;

SELECT provider_id, lab_id, patient_count
FROM lao39.fact_lab f
LEFT JOIN lao39.dim_time d
ON d.date_id = f.date_id
WHERE year=2025 AND month='January'
ORDER BY provider_id, lab_id;

/* emr and fact_procedure */
SELECT provider_id, procedure_id, COUNT(p.visit_id) AS visit_count
FROM emr.visit v
LEFT JOIN emr.visit_procedure p
ON v.visit_id = p.visit_id
WHERE EXTRACT(YEAR FROM visit_date) = 2024 AND EXTRACT(MONTH FROM visit_date) = 3 AND procedure_id IS NOT NULL
GROUP BY provider_id, procedure_id
ORDER BY provider_id, procedure_id;

SELECT provider_id, procedure_id, visit_count
FROM lao39.fact_procedure f
LEFT JOIN lao39.dim_time d
ON d.date_id = f.date_id
WHERE year=2024 AND month='March'
ORDER BY provider_id, procedure_id;

/* emr and fact_diagnosis */
SELECT provider_id, diagnosis_id, COUNT(p.visit_id) AS diagnosis_count
FROM emr.visit v
LEFT JOIN emr.visit_diagnosis p
ON v.visit_id = p.visit_id
WHERE EXTRACT(YEAR FROM visit_date) = 2024 AND EXTRACT(MONTH FROM visit_date) = 1 AND diagnosis_id IS NOT NULL
GROUP BY provider_id, diagnosis_id
ORDER BY provider_id, diagnosis_id;

SELECT provider_id, diagnosis_id, visit_count
FROM lao39.fact_diagnosis f
LEFT JOIN lao39.dim_time d
ON d.date_id = f.date_id
WHERE year=2024 AND month='January'
ORDER BY provider_id, diagnosis_id;

/*
No nulls in surrogate key fields
*/
SELECT COUNT(date_id) 
FROM fact_patient
WHERE date_id IS NULL;

SELECT COUNT(date_id) 
FROM fact_lab
WHERE date_id IS NULL;

SELECT COUNT(date_id) 
FROM fact_procedure
WHERE date_id IS NULL;

SELECT COUNT(date_id) 
FROM fact_diagnosis
WHERE date_id IS NULL;

SELECT COUNT(date_id) 
FROM dim_time
WHERE date_id IS NULL;

/*
Consistent Date Mappings
*/
WITH vtime AS (
	SELECT v.visit_id, v.patient_id, v.provider_id, t.date_id
		,v.visit_date, t.start_dt, t.end_dt
	FROM emr.visit v
	LEFT JOIN lao39.dim_time t
	ON v.visit_date >= t.start_dt AND v.visit_date <= t.end_dt
)
,vcnt AS (
	SELECT t.patient_id, t.provider_id, t.date_id, COUNT(visit_id) AS visit_count
	FROM vtime t
    GROUP BY t.patient_id, t.provider_id, t.date_id
    ORDER BY t.patient_id, t.provider_id, t.date_id
)
SELECT * FROM vcnt;

SELECT t.date_id, patient_id, provider_id, v.visit_date, t.start_dt, t.end_dt
FROM dim_time t
CROSS JOIN emr.visit v
ON v.visit_date >= t.start_dt AND v.visit_date <= t.end_dt; 
