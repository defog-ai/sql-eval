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
-- Name: author; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.author (
    authorid bigint NOT NULL,
    authorname text
);


ALTER TABLE public.author OWNER TO postgres;

--
-- Name: cite; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cite (
    citingpaperid bigint NOT NULL,
    citedpaperid bigint NOT NULL
);


ALTER TABLE public.cite OWNER TO postgres;

--
-- Name: dataset; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dataset (
    datasetid bigint NOT NULL,
    datasetname text
);


ALTER TABLE public.dataset OWNER TO postgres;

--
-- Name: field; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.field (
    fieldid bigint
);


ALTER TABLE public.field OWNER TO postgres;

--
-- Name: journal; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.journal (
    journalid bigint NOT NULL,
    journalname text
);


ALTER TABLE public.journal OWNER TO postgres;

--
-- Name: keyphrase; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.keyphrase (
    keyphraseid bigint NOT NULL,
    keyphrasename text
);


ALTER TABLE public.keyphrase OWNER TO postgres;

--
-- Name: paper; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.paper (
    paperid bigint NOT NULL,
    title text,
    venueid bigint,
    year bigint,
    numciting bigint,
    numcitedby bigint,
    journalid bigint
);


ALTER TABLE public.paper OWNER TO postgres;

--
-- Name: paperdataset; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.paperdataset (
    paperid bigint,
    datasetid bigint
);


ALTER TABLE public.paperdataset OWNER TO postgres;

--
-- Name: paperfield; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.paperfield (
    fieldid bigint,
    paperid bigint
);


ALTER TABLE public.paperfield OWNER TO postgres;

--
-- Name: paperkeyphrase; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.paperkeyphrase (
    paperid bigint,
    keyphraseid bigint
);


ALTER TABLE public.paperkeyphrase OWNER TO postgres;

--
-- Name: venue; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.venue (
    venueid bigint NOT NULL,
    venuename text
);


ALTER TABLE public.venue OWNER TO postgres;

--
-- Name: writes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.writes (
    paperid bigint,
    authorid bigint
);


ALTER TABLE public.writes OWNER TO postgres;

--
-- Data for Name: author; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.author (authorid, authorname) FROM stdin;
1	John Smith
2	Emily Johnson
3	Michael Brown
4	Sarah Davis
5	David Wilson
\.


--
-- Data for Name: cite; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cite (citingpaperid, citedpaperid) FROM stdin;
1	2
2	3
3	4
4	5
5	1
3	5
4	2
1	4
3	1
\.


--
-- Data for Name: dataset; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dataset (datasetid, datasetname) FROM stdin;
1	COVID-19 Research
2	Machine Learning Datasets
3	Climate Change Data
4	Social Media Analysis
\.


--
-- Data for Name: field; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.field (fieldid) FROM stdin;
1
2
3
4
\.


--
-- Data for Name: journal; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.journal (journalid, journalname) FROM stdin;
1	Nature
2	Science
3	IEEE Transactions on Pattern Analysis and Machine Intelligence
4	International Journal of Mental Health
\.


--
-- Data for Name: keyphrase; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.keyphrase (keyphraseid, keyphrasename) FROM stdin;
1	Machine Learning
2	Climate Change
3	Social Media
4	COVID-19
5	Mental Health
\.


--
-- Data for Name: paper; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.paper (paperid, title, venueid, year, numciting, numcitedby, journalid) FROM stdin;
1	A Study on Machine Learning Algorithms	1	2020	2	2	3
2	The Effects of Climate Change on Agriculture	1	2020	1	2	1
3	Social Media and Mental Health	2	2019	3	1	4
4	COVID-19 Impact on Society	1	2020	2	2	2
5	Machine Learning in Tackling Climate Change	2	2019	1	2	3
\.

--
-- Data for Name: paperdataset; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.paperdataset (paperid, datasetid) FROM stdin;
1	2
2	3
3	4
4	1
5	2
5	3
\.


--
-- Data for Name: paperfield; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.paperfield (fieldid, paperid) FROM stdin;
1	1
2	2
3	3
4	4
1	5
\.


--
-- Data for Name: paperkeyphrase; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.paperkeyphrase (paperid, keyphraseid) FROM stdin;
1	1
2	2
3	3
3	5
4	4
5	1
5	2
\.


--
-- Data for Name: venue; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.venue (venueid, venuename) FROM stdin;
1	Conference on Machine Learning
2	International Journal of Climate Change
3	Social Media Analysis Workshop
\.


--
-- Data for Name: writes; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.writes (paperid, authorid) FROM stdin;
1	1
2	2
3	3
4	4
5	5
1	3
1	4
2	3
4	5
5	1
2	1
4	3
\.


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

