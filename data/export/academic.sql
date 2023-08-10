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
    aid bigint NOT NULL,
    homepage text,
    name text,
    oid bigint
);


ALTER TABLE public.author OWNER TO postgres;

--
-- Name: cite; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cite (
    cited bigint,
    citing bigint
);


ALTER TABLE public.cite OWNER TO postgres;

--
-- Name: conference; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.conference (
    cid bigint NOT NULL,
    homepage text,
    name text
);


ALTER TABLE public.conference OWNER TO postgres;

--
-- Name: domain; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.domain (
    did bigint NOT NULL,
    name text
);


ALTER TABLE public.domain OWNER TO postgres;

--
-- Name: domain_author; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.domain_author (
    aid bigint NOT NULL,
    did bigint NOT NULL
);


ALTER TABLE public.domain_author OWNER TO postgres;

--
-- Name: domain_conference; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.domain_conference (
    cid bigint NOT NULL,
    did bigint NOT NULL
);


ALTER TABLE public.domain_conference OWNER TO postgres;

--
-- Name: domain_journal; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.domain_journal (
    did bigint NOT NULL,
    jid bigint NOT NULL
);


ALTER TABLE public.domain_journal OWNER TO postgres;

--
-- Name: domain_keyword; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.domain_keyword (
    did bigint NOT NULL,
    kid bigint NOT NULL
);


ALTER TABLE public.domain_keyword OWNER TO postgres;

--
-- Name: domain_publication; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.domain_publication (
    did bigint NOT NULL,
    pid bigint NOT NULL
);


ALTER TABLE public.domain_publication OWNER TO postgres;

--
-- Name: journal; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.journal (
    homepage text,
    jid bigint NOT NULL,
    name text
);


ALTER TABLE public.journal OWNER TO postgres;

--
-- Name: keyword; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.keyword (
    keyword text,
    kid bigint NOT NULL
);


ALTER TABLE public.keyword OWNER TO postgres;

--
-- Name: organization; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.organization (
    continent text,
    homepage text,
    name text,
    oid bigint NOT NULL
);


ALTER TABLE public.organization OWNER TO postgres;

--
-- Name: publication; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.publication (
    abstract text,
    cid bigint,
    citation_num bigint,
    jid bigint,
    pid bigint NOT NULL,
    reference_num bigint,
    title text,
    year bigint
);


ALTER TABLE public.publication OWNER TO postgres;

--
-- Name: publication_keyword; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.publication_keyword (
    pid bigint NOT NULL,
    kid bigint NOT NULL
);


ALTER TABLE public.publication_keyword OWNER TO postgres;

--
-- Name: writes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.writes (
    aid bigint NOT NULL,
    pid bigint NOT NULL
);


ALTER TABLE public.writes OWNER TO postgres;

--
-- Data for Name: author; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.author (aid, homepage, name, oid) FROM stdin;
1	www.larry.com	Larry Summers	2
2	www.ashish.com	Ashish Vaswani	3
3	www.noam.com	Noam Shazeer	3
4	www.martin.com	Martin Odersky	4
5	\N	Kempinski	\N
\.


--
-- Data for Name: cite; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cite (cited, citing) FROM stdin;
1	2
4	3
5	4
5	3
\.


--
-- Data for Name: conference; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.conference (cid, homepage, name) FROM stdin;
1	www.isa.com	ISA
2	www.aaas.com	AAAS
3	www.icml.com	ICML
\.


--
-- Data for Name: domain; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.domain (did, name) FROM stdin;
1	Data Science
2	Natural Sciences
3	Computer Science
4	Sociology
5	Machine Learning
\.


--
-- Data for Name: domain_author; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.domain_author (aid, did) FROM stdin;
1	2
1	4
2	3
2	1
2	5
3	5
3	3
4	3
\.


--
-- Data for Name: domain_conference; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.domain_conference (cid, did) FROM stdin;
1	2
2	4
3	5
\.


--
-- Data for Name: domain_journal; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.domain_journal (did, jid) FROM stdin;
1	2
2	3
5	4
\.


--
-- Data for Name: domain_keyword; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.domain_keyword (did, kid) FROM stdin;
1	2
2	3
\.


--
-- Data for Name: domain_publication; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.domain_publication (did, pid) FROM stdin;
4	1
2	2
1	3
3	4
3	5
5	5
\.


--
-- Data for Name: journal; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.journal (homepage, jid, name) FROM stdin;
www.aijournal.com	1	Journal of Artificial Intelligence Research
www.nature.com	2	Nature
www.science.com	3	Science
www.ml.com	4	Journal of Machine Learning Research
\.


--
-- Data for Name: keyword; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.keyword (keyword, kid) FROM stdin;
AI	1
Neuroscience	2
Machine Learning	3
Keyword 4	4
\.


--
-- Data for Name: organization; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.organization (continent, homepage, name, oid) FROM stdin;
Asia	www.organization1.com	Organization 1	1
North America	www.organization2.com	Organization 2	2
North America	www.organization3.com	Organization 3	3
Europe	www.epfl.com	École Polytechnique Fédérale de Lausanne 4	4
Europe	www.organization5.com	Organization 5	5
\.


--
-- Data for Name: publication; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.publication (abstract, cid, citation_num, jid, pid, reference_num, title, year) FROM stdin;
Abstract 1	1	10	1	1	5	The Effects of Climate Change on Agriculture	2020
Abstract 2	2	12	2	2	8	A Study on the Effects of Social Media on Mental Health	2020
Abstract 3	3	23	2	3	2	Data Mining Techniques	2021
Abstract 4	3	14	2	4	14	Optimizing GPU Throughput	2021
Abstract 5	3	30	4	5	4	Attention is all you need	2021
\.


--
-- Data for Name: publication_keyword; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.publication_keyword (pid, kid) FROM stdin;
1	2
2	3
\.


--
-- Data for Name: writes; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.writes (aid, pid) FROM stdin;
1	1
1	2
2	3
2	4
2	5
3	5
\.


--
-- Name: author author_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.author
    ADD CONSTRAINT author_pkey PRIMARY KEY (aid);


--
-- Name: conference conference_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.conference
    ADD CONSTRAINT conference_pkey PRIMARY KEY (cid);


--
-- Name: domain_author domain_author_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_author
    ADD CONSTRAINT domain_author_pkey PRIMARY KEY (did, aid);


--
-- Name: domain_conference domain_conference_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_conference
    ADD CONSTRAINT domain_conference_pkey PRIMARY KEY (did, cid);


--
-- Name: domain_journal domain_journal_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_journal
    ADD CONSTRAINT domain_journal_pkey PRIMARY KEY (did, jid);


--
-- Name: domain_keyword domain_keyword_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_keyword
    ADD CONSTRAINT domain_keyword_pkey PRIMARY KEY (did, kid);


--
-- Name: domain domain_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain
    ADD CONSTRAINT domain_pkey PRIMARY KEY (did);


--
-- Name: domain_publication domain_publication_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_publication
    ADD CONSTRAINT domain_publication_pkey PRIMARY KEY (did, pid);


--
-- Name: journal journal_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.journal
    ADD CONSTRAINT journal_pkey PRIMARY KEY (jid);


--
-- Name: keyword keyword_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.keyword
    ADD CONSTRAINT keyword_pkey PRIMARY KEY (kid);


--
-- Name: organization organization_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.organization
    ADD CONSTRAINT organization_pkey PRIMARY KEY (oid);


--
-- Name: publication_keyword publication_keyword_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.publication_keyword
    ADD CONSTRAINT publication_keyword_pkey PRIMARY KEY (kid, pid);


--
-- Name: publication publication_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.publication
    ADD CONSTRAINT publication_pkey PRIMARY KEY (pid);


--
-- Name: writes writes_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.writes
    ADD CONSTRAINT writes_pkey PRIMARY KEY (aid, pid);


--
-- Name: author author_oid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.author
    ADD CONSTRAINT author_oid_fkey FOREIGN KEY (oid) REFERENCES public.organization(oid);


--
-- Name: cite cite_cited_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cite
    ADD CONSTRAINT cite_cited_fkey FOREIGN KEY (cited) REFERENCES public.publication(pid);


--
-- Name: cite cite_citing_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cite
    ADD CONSTRAINT cite_citing_fkey FOREIGN KEY (citing) REFERENCES public.publication(pid);


--
-- Name: domain_author domain_author_aid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_author
    ADD CONSTRAINT domain_author_aid_fkey FOREIGN KEY (aid) REFERENCES public.author(aid);


--
-- Name: domain_author domain_author_did_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_author
    ADD CONSTRAINT domain_author_did_fkey FOREIGN KEY (did) REFERENCES public.domain(did);


--
-- Name: domain_conference domain_conference_cid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_conference
    ADD CONSTRAINT domain_conference_cid_fkey FOREIGN KEY (cid) REFERENCES public.conference(cid);


--
-- Name: domain_conference domain_conference_did_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_conference
    ADD CONSTRAINT domain_conference_did_fkey FOREIGN KEY (did) REFERENCES public.domain(did);


--
-- Name: domain_journal domain_journal_did_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_journal
    ADD CONSTRAINT domain_journal_did_fkey FOREIGN KEY (did) REFERENCES public.domain(did);


--
-- Name: domain_journal domain_journal_jid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_journal
    ADD CONSTRAINT domain_journal_jid_fkey FOREIGN KEY (jid) REFERENCES public.journal(jid);


--
-- Name: domain_keyword domain_keyword_did_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_keyword
    ADD CONSTRAINT domain_keyword_did_fkey FOREIGN KEY (did) REFERENCES public.domain(did);


--
-- Name: domain_keyword domain_keyword_kid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_keyword
    ADD CONSTRAINT domain_keyword_kid_fkey FOREIGN KEY (kid) REFERENCES public.keyword(kid);


--
-- Name: domain_publication domain_publication_did_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_publication
    ADD CONSTRAINT domain_publication_did_fkey FOREIGN KEY (did) REFERENCES public.domain(did);


--
-- Name: domain_publication domain_publication_pid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.domain_publication
    ADD CONSTRAINT domain_publication_pid_fkey FOREIGN KEY (pid) REFERENCES public.publication(pid);


--
-- Name: publication publication_cid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.publication
    ADD CONSTRAINT publication_cid_fkey FOREIGN KEY (cid) REFERENCES public.conference(cid);


--
-- Name: publication publication_jid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.publication
    ADD CONSTRAINT publication_jid_fkey FOREIGN KEY (jid) REFERENCES public.journal(jid);


--
-- Name: publication_keyword publication_keyword_kid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.publication_keyword
    ADD CONSTRAINT publication_keyword_kid_fkey FOREIGN KEY (kid) REFERENCES public.keyword(kid);


--
-- Name: publication_keyword publication_keyword_pid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.publication_keyword
    ADD CONSTRAINT publication_keyword_pid_fkey FOREIGN KEY (pid) REFERENCES public.publication(pid);


--
-- Name: writes writes_aid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.writes
    ADD CONSTRAINT writes_aid_fkey FOREIGN KEY (aid) REFERENCES public.author(aid);


--
-- Name: writes writes_pid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.writes
    ADD CONSTRAINT writes_pid_fkey FOREIGN KEY (pid) REFERENCES public.publication(pid);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

