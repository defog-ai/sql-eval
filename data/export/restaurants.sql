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
-- Name: geographic; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.geographic (
    city_name text,
    county text,
    region text
);


ALTER TABLE public.geographic OWNER TO postgres;

--
-- Name: location; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.location (
    restaurant_id bigint,
    house_number bigint,
    street_name text,
    city_name text
);


ALTER TABLE public.location OWNER TO postgres;

--
-- Name: restaurant; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.restaurant (
    id bigint,
    name text,
    food_type text,
    city_name text,
    rating real
);


ALTER TABLE public.restaurant OWNER TO postgres;

--
-- Data for Name: GEOGRAPHIC; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.geographic (city_name, county, region) FROM stdin;
Los Angeles	Los Angeles	California
New York	New York	New York
San Francisco	San Francisco	California
Miami	Miami-Dade	Florida
Chicago	Cook	Illinois
\.


--
-- Data for Name: LOCATION; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.location (restaurant_id, house_number, street_name, city_name) FROM stdin;
1	123	Main St	Los Angeles
2	456	Maple Ave	Los Angeles
3	789	Oak St	Los Angeles
4	321	Elm St	New York
5	654	Pine Ave	New York
6	123	Pine Ave	New York
7	12	Market St	San Francisco
8	34	Mission St	San Francisco
9	56	Valencia St	San Francisco
10	78	Ocean Dr	Miami
11	90	Biscayne Rd	Miami
\.


--
-- Data for Name: RESTAURANT; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.restaurant (id, rating, name, food_type, city_name) FROM stdin;
1	4.5	The Pasta House	Italian	Los Angeles
2	3.8	The Burger Joint	American	Los Angeles
3	4.2	The Sushi Bar	Japanese	Los Angeles
4	4.7	The Pizza Place	Italian	New York
5	3.9	The Steakhouse	American	New York
6	4.3	The Ramen Shop	Japanese	New York
7	4.1	The Tacos & Burritos	Mexican	San Francisco
8	4.6	The Vegan Cafe	Vegan	San Francisco
9	3.7	The BBQ Joint	American	San Francisco
10	4.4	The Seafood Shack	Seafood	Miami
11	4.6	The Seafood Shack	Seafood	Miami
\.

--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

