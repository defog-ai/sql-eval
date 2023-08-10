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
-- Name: business; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.business (
    bid bigint,
    business_id text,
    name text,
    full_address text,
    city text,
    latitude text,
    longitude text,
    review_count bigint,
    is_open bigint,
    rating real,
    state text
);


ALTER TABLE public.business OWNER TO postgres;

--
-- Name: category; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.category (
    id bigint,
    business_id text,
    category_name text
);


ALTER TABLE public.category OWNER TO postgres;

--
-- Name: checkin; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.checkin (
    cid bigint,
    business_id text,
    count bigint,
    day text
);


ALTER TABLE public.checkin OWNER TO postgres;

--
-- Name: neighbourhood; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.neighbourhood (
    id bigint,
    business_id text,
    neighbourhood_name text
);


ALTER TABLE public.neighbourhood OWNER TO postgres;

--
-- Name: review; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.review (
    rid bigint,
    business_id text,
    user_id text,
    rating real,
    text text,
    year bigint,
    month text
);


ALTER TABLE public.review OWNER TO postgres;

--
-- Name: tip; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tip (
    tip_id bigint,
    business_id text,
    text text,
    user_id text,
    likes bigint,
    year bigint,
    month text
);


ALTER TABLE public.tip OWNER TO postgres;

--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    uid bigint,
    user_id text,
    name text
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Data for Name: business; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.business (bid, business_id, name, full_address, city, latitude, longitude, review_count, is_open, rating, state) FROM stdin;
1	abc123	Joe's Pizza	123 Main St	San Francisco	37.7749295	-122.4194155	1	0	4.5	CA
2	def456	Peter's Cafe	456 Elm St	New York	40.712776	-74.005974	2	1	4.55	NY
3	ghi789	Anna's Diner	789 Oak St	Los Angeles	34.052235	-118.243683	2	0	2.55	CA
4	jkl012	Mark's Bistro	012 Maple St	San Francisco	37.7749295	-122.4194155	1	1	4.8	CA
5	mno345	Lily's Bakery	345 Walnut St	New York	40.712776	-74.005974	1	1	4.6	NY
6	xyz123	Izza's Pizza	83 Main St	San Francisco	37.8749295	-122.5194155	1	1	0.5	CA
7	uvw456	Sashay's Cafe	246 Elm St	New York	40.812776	-74.105974	1	1	4.0	NY
\.


--
-- Data for Name: category; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.category (id, business_id, category_name) FROM stdin;
1	abc123	Pizza
2	def456	Cafe
3	ghi789	Diner
4	jkl012	Bistro
5	mno345	Bakery
6	xyz123	Pizza
7	uvw456	Cafe
\.


--
-- Data for Name: checkin; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.checkin (cid, business_id, count, day) FROM stdin;
1	abc123	10	Monday
2	def456	20	Tuesday
3	ghi789	15	Wednesday
4	jkl012	30	Thursday
5	mno345	25	Friday
6	abc123	13	Tuesday
7	def456	14	Wednesday
8	ghi789	8	Thursday
9	jkl012	21	Saturday
10	mno345	24	Friday
11	xyz123	10	Saturday
12	uvw456	2	Monday
\.


--
-- Data for Name: neighbourhood; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.neighbourhood (id, business_id, neighbourhood_name) FROM stdin;
1	abc123	Downtown
2	def456	Midtown
3	ghi789	Hollywood
4	jkl012	Downtown
5	mno345	Upper East Side
6	xyz123	Downtown
7	uvw456	Midtown
\.


--
-- Data for Name: review; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.review (rid, business_id, user_id, rating, text, year, month) FROM stdin;
1	abc123	1	4.5	Great pizza!	2021	January
2	def456	2	4.2	Delicious food.	2021	February
3	ghi789	3	3.9	Average diner.	2021	March
4	jkl012	4	4.8	Amazing bistro.	2021	April
5	mno345	5	4.6	Yummy bakery.	2021	January
6	ghi789	1	1.2	Horrible staff!	2021	April
7	def456	2	4.9	Second visit. I'm loving it.	2021	May
8	xyz123	3	0.5	Hate it	2021	June
9	uvw456	4	4.0	Not bad.	2021	July
\.


--
-- Data for Name: tip; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.tip (tip_id, business_id, text, user_id, likes, year, month) FROM stdin;
1	abc123	Try their pepperoni pizza!	1	\N	2021	January
2	def456	Their coffee is amazing.	2	\N	2021	February
3	ghi789	The pancakes are delicious.	3	\N	2021	March
4	jkl012	Highly recommend the steak.	4	\N	2021	April
5	mno345	Their pastries are to die for.	5	\N	2021	May
6	xyz123	Don't waste your money.	1	\N	2021	June
7	uvw456	Not bad.	2	\N	2021	July
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (uid, user_id, name) FROM stdin;
1	1	John Doe
2	2	Jane Smith
3	3	David Johnson
4	4	Sarah Williams
5	5	Michael Brown
\.


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

