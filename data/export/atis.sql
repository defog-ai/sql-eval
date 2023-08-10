--
-- PostgreSQL database dump
--

-- Dumped from database version 14.8
-- Dumped by pg_dump version 15.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: postgres
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: aircraft; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.aircraft (
    aircraft_code text,
    aircraft_description text,
    manufacturer text,
    basic_type text,
    engines bigint,
    propulsion text,
    wide_body text,
    wing_span bigint,
    length bigint,
    weight bigint,
    capacity bigint,
    pay_load bigint,
    cruising_speed bigint,
    range_miles bigint,
    pressurized text
);


ALTER TABLE public.aircraft OWNER TO postgres;

--
-- Name: airline; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.airline (
    airline_code text,
    airline_name text,
    note text
);


ALTER TABLE public.airline OWNER TO postgres;

--
-- Name: airport; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.airport (
    airport_code text,
    airport_name text,
    airport_location text,
    state_code text,
    country_name text,
    time_zone_code text,
    minimum_connect_time bigint
);


ALTER TABLE public.airport OWNER TO postgres;

--
-- Name: airport_service; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.airport_service (
    city_code text,
    airport_code text,
    miles_distant bigint,
    direction text,
    minutes_distant bigint
);


ALTER TABLE public.airport_service OWNER TO postgres;

--
-- Name: city; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.city (
    city_code text,
    city_name text,
    state_code text,
    country_name text,
    time_zone_code text
);


ALTER TABLE public.city OWNER TO postgres;

--
-- Name: class_of_service; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.class_of_service (
    booking_class text DEFAULT ''::text NOT NULL,
    rank bigint,
    class_description text
);


ALTER TABLE public.class_of_service OWNER TO postgres;

--
-- Name: code_description; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.code_description (
    code text DEFAULT ''::text NOT NULL,
    description text
);


ALTER TABLE public.code_description OWNER TO postgres;

--
-- Name: compartment_class; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.compartment_class (
    compartment text,
    class_type text
);


ALTER TABLE public.compartment_class OWNER TO postgres;

--
-- Name: date_day; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.date_day (
    month_number bigint,
    day_number bigint,
    year bigint,
    day_name text
);


ALTER TABLE public.date_day OWNER TO postgres;

--
-- Name: days; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.days (
    days_code text,
    day_name text
);


ALTER TABLE public.days OWNER TO postgres;

--
-- Name: dual_carrier; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dual_carrier (
    main_airline text,
    low_flight_number bigint,
    high_flight_number bigint,
    dual_airline text,
    service_name text
);


ALTER TABLE public.dual_carrier OWNER TO postgres;

--
-- Name: equipment_sequence; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.equipment_sequence (
    aircraft_code_sequence text,
    aircraft_code text
);


ALTER TABLE public.equipment_sequence OWNER TO postgres;

--
-- Name: fare; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fare (
    fare_id bigint DEFAULT '0'::bigint NOT NULL,
    from_airport text,
    to_airport text,
    fare_basis_code text,
    fare_airline text,
    restriction_code text,
    one_direction_cost bigint,
    round_trip_cost bigint,
    round_trip_required text
);


ALTER TABLE public.fare OWNER TO postgres;

--
-- Name: fare_basis; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fare_basis (
    fare_basis_code text,
    booking_class text,
    class_type text,
    premium text,
    economy text,
    discounted text,
    night text,
    season text,
    basis_days text
);


ALTER TABLE public.fare_basis OWNER TO postgres;

--
-- Name: flight; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.flight (
    flight_id bigint DEFAULT '0'::bigint NOT NULL,
    flight_days text,
    from_airport text,
    to_airport text,
    departure_time bigint,
    arrival_time bigint,
    airline_flight text,
    airline_code text,
    flight_number text,
    aircraft_code_sequence text,
    meal_code text,
    stops bigint,
    connections bigint,
    dual_carrier text,
    time_elapsed bigint
);


ALTER TABLE public.flight OWNER TO postgres;

--
-- Name: flight_fare; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.flight_fare (
    flight_id bigint,
    fare_id bigint
);


ALTER TABLE public.flight_fare OWNER TO postgres;

--
-- Name: flight_leg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.flight_leg (
    flight_id bigint,
    leg_number bigint,
    leg_flight bigint
);


ALTER TABLE public.flight_leg OWNER TO postgres;

--
-- Name: flight_stop; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.flight_stop (
    flight_id bigint,
    stop_number bigint,
    stop_days text,
    stop_airport text,
    arrival_time bigint,
    arrival_airline text,
    arrival_flight_number bigint,
    departure_time bigint,
    departure_airline text,
    departure_flight_number bigint,
    stop_time bigint
);


ALTER TABLE public.flight_stop OWNER TO postgres;

--
-- Name: food_service; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.food_service (
    meal_code text,
    meal_number bigint,
    compartment text,
    meal_description text
);


ALTER TABLE public.food_service OWNER TO postgres;

--
-- Name: ground_service; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ground_service (
    city_code text,
    airport_code text,
    transport_type text,
    ground_fare bigint
);


ALTER TABLE public.ground_service OWNER TO postgres;

--
-- Name: month; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.month (
    month_number bigint,
    month_name text
);


ALTER TABLE public.month OWNER TO postgres;

--
-- Name: restriction; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.restriction (
    restriction_code text,
    advance_purchase bigint,
    stopovers text,
    saturday_stay_required text,
    minimum_stay bigint,
    maximum_stay bigint,
    application text,
    no_discounts text
);


ALTER TABLE public.restriction OWNER TO postgres;

--
-- Name: state; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.state (
    state_code text,
    state_name text,
    country_name text
);


ALTER TABLE public.state OWNER TO postgres;

--
-- Name: time_interval; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.time_interval (
    period text,
    begin_time bigint,
    end_time bigint
);


ALTER TABLE public.time_interval OWNER TO postgres;

--
-- Name: time_zone; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.time_zone (
    time_zone_code text,
    time_zone_name text,
    hours_from_gmt bigint
);


ALTER TABLE public.time_zone OWNER TO postgres;

--
-- Data for Name: aircraft; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.aircraft (aircraft_code, aircraft_description, manufacturer, basic_type, engines, propulsion, wide_body, wing_span, length, weight, capacity, pay_load, cruising_speed, range_miles, pressurized) FROM stdin;
B747	The Boeing 747 is a wide-body airliner.	Boeing	\N	\N	\N	\N	\N	\N	\N	10	3	300	10000	\N
A320	The Airbus A320 is a narrow-body airliner.	Airbus	\N	\N	\N	\N	\N	\N	\N	10	4	150	12000	\N
B737	The Boeing 737 is a narrow-body airliner.	Boeing	\N	\N	\N	\N	\N	\N	\N	10	10	350	22000	\N
A380	The Airbus A380 is a wide-body airliner.	Airbus	\N	\N	\N	\N	\N	\N	\N	10	3	350	12000	\N
B777	The Boeing 777 is a wide-body airliner.	Boeing	\N	\N	\N	\N	\N	\N	\N	10	3	200	10000	\N
A330	The Airbus A330 is a wide-body airliner.	Airbus	\N	\N	\N	\N	\N	\N	\N	10	3	200	10000	\N
B787	The Boeing 787 is a wide-body airliner.	Boeing	\N	\N	\N	\N	\N	\N	\N	10	3	200	15000	\N
A350	The Airbus A350 is a wide-body airliner.	Airbus	\N	\N	\N	\N	\N	\N	\N	10	3	270	13000	\N
E190	The Embraer E190 is a narrow-body airliner.	Embraer	\N	\N	\N	\N	\N	\N	\N	10	3	280	13000	\N
CRJ200	The Bombardier CRJ200 is a regional jet.	Bombardier	\N	\N	\N	\N	\N	\N	\N	20	4	200	20000	\N
\.


--
-- Data for Name: airline; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.airline (airline_code, airline_name, note) FROM stdin;
AA	American Airlines	\N
UA	United Airlines	\N
DL	Delta Air Lines	\N
WN	Southwest Airlines	\N
AS	Alaska Airlines	\N
B6	JetBlue Airways	\N
NK	Spirit Airlines	\N
F9	Frontier Airlines	\N
HA	Hawaiian Airlines	\N
VX	Virgin America	\N
\.


--
-- Data for Name: airport; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.airport (airport_code, airport_name, airport_location, state_code, country_name, time_zone_code, minimum_connect_time) FROM stdin;
JFK	John F. Kennedy International Airport	New York City	NY	United States	EST	23
LAX	Los Angeles International Airport	Los Angeles	CA	United States	PST	20
ORD	O'Hare International Airport	Chicago	IL	United States	CST	24
DFW	Dallas/Fort Worth International Airport	Dallas	TX	United States	CST	40
DEN	Denver International Airport	Denver	CO	United States	MST	42
ATL	Hartsfield-Jackson Atlanta International Airport	Atlanta	GA	United States	EST	10
SFO	San Francisco International Airport	San Francisco	CA	United States	PST	49
SEA	Seattle-Tacoma International Airport	Seattle	WA	United States	PST	50
LAS	McCarran International Airport	Las Vegas	NV	United States	PST	30
MCO	Orlando International Airport	Orlando	FL	United States	EST	50
\.


--
-- Data for Name: airport_service; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.airport_service (city_code, airport_code, miles_distant, direction, minutes_distant) FROM stdin;
NYC	JFK	10	North	20
NYC	JFK	20	South	40
NYC	JFK	30	East	60
NYC	JFK	40	West	80
NYC	JFK	50	Northeast	100
NYC	JFK	60	Northwest	120
NYC	JFK	70	Southeast	140
NYC	JFK	80	Southwest	160
NYC	JFK	90	Up	180
NYC	JFK	100	Down	200
\.


--
-- Data for Name: city; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.city (city_code, city_name, state_code, country_name, time_zone_code) FROM stdin;
NYC	New York	NY	United States	EST
LAX	Los Angeles	CA	United States	PST
CHI	Chicago	IL	United States	CST
DAL	Dallas	TX	United States	CST
DEN	Denver	CO	United States	MST
ATL	Atlanta	GA	United States	EST
SFO	San Francisco	CA	United States	PST
SEA	Seattle	WA	United States	PST
LAS	Las Vegas	NV	United States	PST
ORL	Orlando	FL	United States	EST
\.


--
-- Data for Name: class_of_service; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.class_of_service (booking_class, rank, class_description) FROM stdin;
First	1	First Class
Business	2	Business Class
Economy	3	Economy Class
\.


--
-- Data for Name: code_description; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.code_description (code, description) FROM stdin;
ABC	Code ABC
DEF	Code DEF
GHI	Code GHI
JKL	Code JKL
MNO	Code MNO
PQR	Code PQR
STU	Code STU
VWX	Code VWX
YZ	Code YZ
AAA	Code AAA
\.


--
-- Data for Name: compartment_class; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.compartment_class (compartment, class_type) FROM stdin;
First	First Class
Business	Business Class
Economy	Economy Class
\.


--
-- Data for Name: date_day; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.date_day (month_number, day_number, year, day_name) FROM stdin;
\.


--
-- Data for Name: days; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.days (days_code, day_name) FROM stdin;
1	Monday
2	Tuesday
3	Wednesday
4	Thursday
5	Friday
6	Saturday
7	Sunday
\.


--
-- Data for Name: dual_carrier; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dual_carrier (main_airline, low_flight_number, high_flight_number, dual_airline, service_name) FROM stdin;
AA	1	10	VX	Dual Service 1
UA	11	20	DL	Dual Service 2
DL	21	30	UA	Dual Service 3
WN	31	40	AS	Dual Service 4
AS	41	50	WN	Dual Service 5
B6	51	60	NK	Dual Service 6
NK	61	70	B6	Dual Service 7
F9	71	80	HA	Dual Service 8
HA	81	90	F9	Dual Service 9
VX	91	100	AA	Dual Service 10
\.


--
-- Data for Name: equipment_sequence; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.equipment_sequence (aircraft_code_sequence, aircraft_code) FROM stdin;
1	B747
2	A320
3	B737
4	A380
5	B777
6	A330
7	B787
8	A350
9	E190
10	CRJ200
\.


--
-- Data for Name: fare; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.fare (fare_id, from_airport, to_airport, fare_basis_code, fare_airline, restriction_code, one_direction_cost, round_trip_cost, round_trip_required) FROM stdin;
1	JFK	LAX	ABC	AA	NONE	200	300	Yes
2	JFK	ORD	DEF	UA	NONE	150	280	No
3	JFK	ORD	GHI	DL	NONE	180	300	No
6	ORD	JFK	PQR	B6	BLACKOUT	190	350	Yes
7	ORD	JFK	STU	NK	NONE	210	400	Yes
8	ORD	JFK	VWX	F9	NONE	230	400	No
9	LAX	SFO	YZ	HA	NONE	240	400	No
10	ORD	LAX	AAA	VX	NONE	270	500	No
4	LAX	ORD	JKL	WN	NONE	250	350	Yes
5	LAX	ORD	MNO	AS	BLACKOUT	220	400	Yes
\.


--
-- Data for Name: fare_basis; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.fare_basis (fare_basis_code, booking_class, class_type, premium, economy, discounted, night, season, basis_days) FROM stdin;
ABC	First	First Class	Yes	No	No	No	Regular	30
DEF	Business	Business Class	Yes	No	No	No	Regular	30
GHI	Economy	Economy Class	No	Yes	Yes	No	Regular	30
JKL	First	First Class	Yes	No	No	No	Regular	30
MNO	Business	Business Class	Yes	No	No	No	Regular	30
PQR	Economy	Economy Class	No	Yes	Yes	No	Regular	30
STU	First	First Class	Yes	No	No	No	Regular	30
VWX	Business	Business Class	Yes	No	No	No	Regular	30
YZ	Economy	Economy Class	No	Yes	Yes	No	Regular	30
AAA	First	First Class	Yes	No	No	No	Regular	30
\.


--
-- Data for Name: flight; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.flight (flight_id, flight_days, from_airport, to_airport, departure_time, arrival_time, airline_flight, airline_code, flight_number, aircraft_code_sequence, meal_code, stops, connections, dual_carrier, time_elapsed) FROM stdin;
1	mon,wed	ORD	JFK	1577836800	1577840400	AA123	AA	AA123	1	BF	0	0	AA123	3600
2	tue,thu	ORD	JFK	1577844000	1577847700	UA456	UA	UA456	2	LN	1	1	UA456	3700
3	wed	ORD	JFK	1577851200	1577854900	AA789	AA	AA789	3	DN	0	0	AA789	3700
4	thu	ORD	JFK	1577858400	1577862400	WN012	WN	WN012	4	BS	1	1	WN012	4000
5	fri	ORD	LAX	1577865600	1577869600	AS345	AS	AS345	5	BF	0	0	AS345	4000
6	sat,mon	JFK	ORD	1577872800	1577876400	AA124	AA	AA123	6	LN	1	1	B678	3600
7	sun	JFK	ORD	1577880000	1577883700	UA457	UA	UA457	7	DN	0	0	UA457	3700
8	mon	JFK	ORD	1577887200	1577890900	F934	F9	F934	8	BS	1	1	F934	3700
9	tue	LAX	ORD	1577894400	1577898400	HA567	HA	HA567	9	LS	0	0	HA567	4000
10	wed,mon	LAX	ORD	1577901600	1577905600	VX890	VX	VX890	10	DS	1	1	VX890	4000
\.


--
-- Data for Name: flight_fare; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.flight_fare (flight_id, fare_id) FROM stdin;
1	1
2	2
3	3
4	4
5	5
6	6
7	7
8	8
9	9
10	10
\.


--
-- Data for Name: flight_leg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.flight_leg (flight_id, leg_number, leg_flight) FROM stdin;
1	1	1
2	1	2
3	1	3
4	1	4
5	1	5
6	1	6
7	1	7
8	1	8
9	1	9
10	1	10
\.


--
-- Data for Name: flight_stop; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.flight_stop (flight_id, stop_number, stop_days, stop_airport, arrival_time, arrival_airline, arrival_flight_number, departure_time, departure_airline, departure_flight_number, stop_time) FROM stdin;
1	1	1	DFW	1577840400	UA	2	1577836800	AA	1	3600
2	1	2	DFW	1577847600	DL	3	1577844000	UA	2	3600
3	1	3	DEN	1577854800	WN	4	1577851200	DL	3	3600
4	1	4	DEN	1577862000	AS	5	1577858400	WN	4	3600
5	1	5	JFK	1577869200	B6	6	1577865600	AS	5	3600
6	1	6	SFO	1577876400	NK	7	1577872800	B6	6	3600
7	1	7	LAX	1577883600	F9	8	1577880000	NK	7	3600
8	1	1	LAX	1577890800	HA	9	1577887200	F9	8	3600
9	1	2	DFW	1577898000	VX	10	1577894400	HA	9	3600
10	1	3	JFK	1577905200	AA	1	1577901600	VX	10	3600
\.


--
-- Data for Name: food_service; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.food_service (meal_code, meal_number, compartment, meal_description) FROM stdin;
BF	1	First Class	Breakfast
LN	2	First Class	Lunch
DN	3	First Class	Dinner
BS	4	Economy	Breakfast
LS	5	Economy	Lunch
DS	6	Economy	Dinner
\.


--
-- Data for Name: ground_service; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ground_service (city_code, airport_code, transport_type, ground_fare) FROM stdin;
NYC	JFK	Taxi	50
NYC	JFK	Shuttle	40
NYC	JFK	Bus	30
NYC	JFK	Car Rental	60
NYC	JFK	Limousine	70
NYC	JFK	Train	80
NYC	JFK	Subway	90
NYC	JFK	Private Car	100
NYC	JFK	Shared Ride	110
NYC	JFK	Helicopter	120
\.


--
-- Data for Name: month; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.month (month_number, month_name) FROM stdin;
1	January
2	February
3	March
4	April
5	May
6	June
7	July
8	August
9	September
10	October
\.


--
-- Data for Name: restriction; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.restriction (restriction_code, advance_purchase, stopovers, saturday_stay_required, minimum_stay, maximum_stay, application, no_discounts) FROM stdin;
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
NONE	14	2	No	7	30	One-Way	Yes
\.


--
-- Data for Name: state; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.state (state_code, state_name, country_name) FROM stdin;
NY	New York	United States
CA	California	United States
IL	Illinois	United States
TX	Texas	United States
CO	Colorado	United States
GA	Georgia	United States
WA	Washington	United States
NV	Nevada	United States
FL	Florida	United States
\.


--
-- Data for Name: time_interval; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.time_interval (period, begin_time, end_time) FROM stdin;
daily	1577836800	1577840400
daily	1577844000	1577847600
daily	1577851200	1577854800
daily	1577858400	1577862000
daily	1577865600	1577869200
daily	1577872800	1577876400
daily	1577880000	1577883600
daily	1577887200	1577890800
daily	1577894400	1577898000
daily	1577901600	1577905200
\.


--
-- Data for Name: time_zone; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.time_zone (time_zone_code, time_zone_name, hours_from_gmt) FROM stdin;
\.


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

