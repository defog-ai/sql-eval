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
-- Name: border_info; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.border_info (
    state_name text,
    border text
);


ALTER TABLE public.border_info OWNER TO postgres;

--
-- Name: city; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.city (
    city_name text,
    population bigint,
    country_name text DEFAULT ''::text NOT NULL,
    state_name text
);


ALTER TABLE public.city OWNER TO postgres;

--
-- Name: highlow; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.highlow (
    state_name text,
    highest_elevation text,
    lowest_point text,
    highest_point text,
    lowest_elevation text
);


ALTER TABLE public.highlow OWNER TO postgres;

--
-- Name: lake; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lake (
    lake_name text,
    area double precision,
    country_name text DEFAULT ''::text NOT NULL,
    state_name text
);


ALTER TABLE public.lake OWNER TO postgres;

--
-- Name: mountain; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mountain (
    mountain_name text,
    mountain_altitude bigint,
    country_name text DEFAULT ''::text NOT NULL,
    state_name text
);


ALTER TABLE public.mountain OWNER TO postgres;

--
-- Name: river; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.river (
    river_name text,
    length bigint,
    country_name text DEFAULT ''::text NOT NULL,
    traverse text
);


ALTER TABLE public.river OWNER TO postgres;

--
-- Name: state; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.state (
    state_name text,
    population bigint,
    area double precision,
    country_name text DEFAULT ''::text NOT NULL,
    capital text,
    density double precision
);


ALTER TABLE public.state OWNER TO postgres;

--
-- Data for Name: border_info; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.border_info (state_name, border) FROM stdin;
California	Nevada
California	Arizona
California	Oregon
Texas	Louisiana
Texas	Oklahoma
Texas	New Mexico
Florida	Alabama
Florida	Georgia
Florida	Atlantic Ocean
New York	Pennsylvania
New York	Connecticut
New York	Massachusetts
\.


--
-- Data for Name: city; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.city (city_name, population, country_name, state_name) FROM stdin;
New York	1000000	United States	New York
Los Angeles	5000000	United States	California
Chicago	1500000	United States	Illinois
Houston	2000000	United States	Texas
Toronto	800000	Canada	Ontario
Mexico City	600000	Mexico	Distrito Federal
Sao Paulo	3000000	Brazil	Sao Paulo
Mumbai	1200000	India	Maharashtra
London	900000	United Kingdom	England
Tokyo	700000	Japan	Tokyo
\.


--
-- Data for Name: highlow; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.highlow (state_name, highest_elevation, lowest_point, highest_point, lowest_elevation) FROM stdin;
California	4421	Death Valley	Mount Whitney	-86
Texas	2667	Gulf of Mexico	Guadalupe Peak	0
Florida	None	Atlantic Ocean	Unnamed location	0
New York	1629	Atlantic Ocean	Mount Marcy	0
Ontario	None	Atlantic Ocean	Unnamed location	0
Sao Paulo	None	Atlantic Ocean	Unnamed location	0
Guangdong	None	South China Sea	Unnamed location	0
Maharashtra	None	Arabian Sea	Unnamed location	0
England	978	North Sea	Scafell Pike	0
Tokyo	3776	Pacific Ocean	Mount Fuji	0
\.


--
-- Data for Name: lake; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.lake (lake_name, area, country_name, state_name) FROM stdin;
Lake Superior	1000	United States	Michigan
Lake Michigan	500	United States	Michigan
Lake Huron	300	United States	Michigan
Lake Erie	200	United States	Ohio
Lake Ontario	400	United States	New York
Lake Victoria	800	Tanzania	\N
Lake Tanganyika	600	Tanzania	\N
Lake Malawi	700	Tanzania	\N
Lake Baikal	900	Russia	\N
Lake Qinghai	1200	China	\N
\.


--
-- Data for Name: mountain; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.mountain (mountain_name, mountain_altitude, country_name, state_name) FROM stdin;
Mount Everest	10000	Nepal	\N
K2	5000	Pakistan	\N
Kangchenjunga	3000	Nepal	\N
Lhotse	2000	Nepal	\N
Makalu	4000	Nepal	\N
Cho Oyu	8000	Nepal	\N
Dhaulagiri	6000	Nepal	\N
Manaslu	7000	Nepal	\N
Nanga Parbat	9000	Pakistan	\N
Annapurna	1000	Nepal	\N
\.


--
-- Data for Name: river; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.river (river_name, length, country_name, traverse) FROM stdin;
Nile	1000	Egypt	Cairo,Luxor,Aswan
Amazon	500	Brazil	Manaus,Belem
Yangtze	300	China	Shanghai,Wuhan,Chongqing
Mississippi	200	United States	New Orleans,Memphis,St. Louis
Yukon	400	Canada	Whitehorse,Dawson City
Volga	800	Russia	Moscow,Samara,Kazan
Mekong	600	Vietnam	Ho Chi Minh City,Phnom Penh
Danube	700	Germany	Passau,Vienna,Budapest
Rhine	900	Germany	Strasbourg,Frankfurt,Cologne
Po	100	Italy	Turin,Milan,Venice
\.


--
-- Data for Name: state; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.state (state_name, population, area, country_name, capital, density) FROM stdin;
California	100000	10000	United States	Sacramento	1000
Texas	50000	5000	United States	Austin	1000
Florida	150000	15000	United States	Tallahassee	1000
New York	200000	20000	United States	Albany	1000
Ontario	80000	8000	Canada	Toronto	1000
Sao Paulo	50000	6000	Brazil	Sao Paulo	1000
Guangdong	200000	30000	China	Guangzhou	1000
Maharashtra	200000	12000	India	Mumbai	1000
England	9000	10000	United Kingdom	London	1000
Tokyo	70000	50000	Japan	Tokyo	1000
Ohio	90000	11000	United States	Columbus	1000
Michigan	120000	9000	United States	Lansing	1000
\.


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

