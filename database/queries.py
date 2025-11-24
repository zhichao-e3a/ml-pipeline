HISTORICAL = """
SELECT
uu.mobile,
r.id,
r.start_ts,
r.start_test_ts,
r.contraction_url,
r.hb_baby_url,
-- r.raw_fetal_url,
r.basic_info,
r.conclusion,
tt.expected_born_date,
tt.end_born_ts,
r.utime
FROM
extant_future_user.user AS uu
INNER JOIN
extant_future_data.origin_data_record AS r
ON uu.id = r.user_id
INNER JOIN
extant_future_user.user_detail AS tt
ON uu.id = tt.uid
AND r.contraction_url <> ''
AND r.hb_baby_url <> ''
AND tt.end_born_ts IS NOT NULL
AND tt.end_born_ts <> 0
AND r.utime > '{last_utime}'
;
"""

RECRUITED = """
SELECT
uu.mobile,
r.id,
r.start_ts,
r.start_test_ts,
r.contraction_url,
r.hb_baby_url,
-- r.raw_fetal_url,
r.basic_info,
r.conclusion
FROM
extant_future_user.user AS uu
INNER JOIN
origin_data_record AS r
ON uu.id = r.user_id
AND r.contraction_url <> ''
AND r.hb_baby_url <> ''
AND uu.mobile IN ({numbers})
AND r.start_ts BETWEEN UNIX_TIMESTAMP({start}) AND UNIX_TIMESTAMP({end})
;
"""

RECRUITED_PATIENTS_QUERY = """
SELECT
u.mobile,
FROM_UNIXTIME(r.start_ts) AS m_time,
r.basic_info,
r.conclusion
FROM extant_future_user.user AS u
JOIN extant_future_data.origin_data_record AS r
ON u.id = r.user_id
WHERE
u.mobile
IN
({mobile_query_str})
;
"""

HISTORICAL_PATIENTS_QUERY = """
SELECT
uu.name,
u.mobile,
uu.age,
oo.earliest,
oo.latest,
oo.basic_info,
oo.conclusion,
uu.height,
uu.old_weight,
uu.expected_born_date AS edd,
mm.record_type,
mm.record_answer
FROM extant_future_user.user AS u
JOIN extant_future_user.user_detail AS uu ON u.id = uu.uid
JOIN
(
	SELECT
	o1.user_id,
	FROM_UNIXTIME(o1.earliest) AS earliest,
	FROM_UNIXTIME(o1.latest) AS latest,
	o2.basic_info,
	o2.conclusion
	FROM
	(
		SELECT
		user_id,
		MIN(start_ts) AS earliest,
		MAX(start_ts) AS latest
		FROM extant_future_data.origin_data_record
		GROUP BY user_id
	) AS o1
	JOIN
	(
		SELECT user_id, start_ts, basic_info, conclusion
		FROM extant_future_data.origin_data_record
	) AS o2
	ON o1.user_id = o2.user_id AND o1.earliest = o2.start_ts
) AS oo ON uu.uid = oo.user_id
LEFT JOIN extant_future_user.medical_record AS mm ON oo.user_id = mm.user_id AND mm.record_type IN (1, 2, 4, 5, 8, 13)
WHERE
u.mobile
IN
({mobile_query_str})
;
"""