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
-- Name: area; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.area (
    course_id bigint,
    area text
);


ALTER TABLE public.area OWNER TO postgres;

--
-- Name: comment_instructor; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.comment_instructor (
    instructor_id bigint DEFAULT '0'::bigint NOT NULL,
    student_id bigint DEFAULT '0'::bigint NOT NULL,
    score bigint,
    comment_text text
);


ALTER TABLE public.comment_instructor OWNER TO postgres;

--
-- Name: course; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.course (
    course_id bigint DEFAULT '0'::bigint NOT NULL,
    name text,
    department text,
    number text,
    credits text,
    advisory_requirement text,
    enforced_requirement text,
    description text,
    num_semesters bigint,
    num_enrolled bigint,
    has_discussion boolean,
    has_lab boolean,
    has_projects boolean,
    has_exams boolean,
    num_reviews bigint,
    clarity_score bigint,
    easiness_score bigint,
    helpfulness_score bigint
);


ALTER TABLE public.course OWNER TO postgres;

--
-- Name: course_offering; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.course_offering (
    offering_id bigint DEFAULT '0'::bigint NOT NULL,
    course_id bigint,
    semester bigint,
    section_number bigint,
    start_time time without time zone,
    end_time time without time zone,
    monday text,
    tuesday text,
    wednesday text,
    thursday text,
    friday text,
    saturday text,
    sunday text,
    has_final_project boolean,
    has_final_exam boolean,
    textbook text,
    class_address text,
    allow_audit text DEFAULT 'n'::text
);


ALTER TABLE public.course_offering OWNER TO postgres;

--
-- Name: course_prerequisite; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.course_prerequisite (
    pre_course_id bigint NOT NULL,
    course_id bigint NOT NULL
);


ALTER TABLE public.course_prerequisite OWNER TO postgres;

--
-- Name: course_tags_count; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.course_tags_count (
    course_id bigint DEFAULT '0'::bigint NOT NULL,
    clear_grading bigint DEFAULT '0'::bigint,
    pop_quiz bigint DEFAULT '0'::bigint,
    group_projects bigint DEFAULT '0'::bigint,
    inspirational bigint DEFAULT '0'::bigint,
    long_lectures bigint DEFAULT '0'::bigint,
    extra_credit bigint DEFAULT '0'::bigint,
    few_tests bigint DEFAULT '0'::bigint,
    good_feedback bigint DEFAULT '0'::bigint,
    tough_tests bigint DEFAULT '0'::bigint,
    heavy_papers bigint DEFAULT '0'::bigint,
    cares_for_students bigint DEFAULT '0'::bigint,
    heavy_assignments bigint DEFAULT '0'::bigint,
    respected bigint DEFAULT '0'::bigint,
    participation bigint DEFAULT '0'::bigint,
    heavy_reading bigint DEFAULT '0'::bigint,
    tough_grader bigint DEFAULT '0'::bigint,
    hilarious bigint DEFAULT '0'::bigint,
    would_take_again bigint DEFAULT '0'::bigint,
    good_lecture bigint DEFAULT '0'::bigint,
    no_skip bigint DEFAULT '0'::bigint
);


ALTER TABLE public.course_tags_count OWNER TO postgres;

--
-- Name: gsi; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gsi (
    course_offering_id bigint DEFAULT '0'::bigint NOT NULL,
    student_id bigint NOT NULL
);


ALTER TABLE public.gsi OWNER TO postgres;

--
-- Name: instructor; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.instructor (
    instructor_id bigint DEFAULT '0'::bigint NOT NULL,
    name text,
    uniqname text
);


ALTER TABLE public.instructor OWNER TO postgres;

--
-- Name: offering_instructor; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.offering_instructor (
    offering_instructor_id bigint DEFAULT '0'::bigint NOT NULL,
    offering_id bigint,
    instructor_id bigint
);


ALTER TABLE public.offering_instructor OWNER TO postgres;

--
-- Name: program; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.program (
    program_id bigint NOT NULL,
    name text,
    college text,
    introduction text
);


ALTER TABLE public.program OWNER TO postgres;

--
-- Name: program_course; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.program_course (
    program_id bigint DEFAULT '0'::bigint NOT NULL,
    course_id bigint DEFAULT '0'::bigint NOT NULL,
    workload bigint,
    category text DEFAULT ''::text NOT NULL
);


ALTER TABLE public.program_course OWNER TO postgres;

--
-- Name: program_requirement; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.program_requirement (
    program_id bigint NOT NULL,
    category text NOT NULL,
    min_credit bigint,
    additional_req text
);


ALTER TABLE public.program_requirement OWNER TO postgres;

--
-- Name: semester; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.semester (
    semester_id bigint NOT NULL,
    semester text,
    year bigint
);


ALTER TABLE public.semester OWNER TO postgres;

--
-- Name: student; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.student (
    student_id bigint NOT NULL,
    lastname text,
    firstname text,
    program_id bigint,
    declare_major text,
    total_credit bigint,
    total_gpa numeric,
    entered_as text DEFAULT 'firstyear'::text,
    admit_term date,
    predicted_graduation_semester date,
    degree text,
    minor text,
    internship text
);


ALTER TABLE public.student OWNER TO postgres;

--
-- Name: student_record; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.student_record (
    student_id bigint NOT NULL,
    course_id bigint NOT NULL,
    semester bigint NOT NULL,
    grade text,
    how text,
    transfer_source text,
    earn_credit text DEFAULT 'y'::text NOT NULL,
    repeat_term text,
    test_id text,
    offering_id bigint
);


ALTER TABLE public.student_record OWNER TO postgres;

--
-- Data for Name: area; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.area (course_id, area) FROM stdin;
1	Computer Science
2	Mathematics
3	Physics
4	Computer Science
\.


--
-- Data for Name: comment_instructor; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.comment_instructor (instructor_id, student_id, score, comment_text) FROM stdin;
1	1	5	John Smith is a great instructor.
2	2	4	Jane Doe explains concepts clearly.
\.


--
-- Data for Name: course; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.course (course_id, name, department, number, credits, advisory_requirement, enforced_requirement, description, num_semesters, num_enrolled, has_discussion, has_lab, has_projects, has_exams, num_reviews, clarity_score, easiness_score, helpfulness_score) FROM stdin;
1	Introduction to Computer Science	Computer Science	CS101	3	\N	\N	This course introduces the basics of computer science.	2	2	true	false	true	false	10	5	3	4
2	Advanced Calculus	Mathematics	MATH201	4	CS101	\N	This course covers advanced topics in calculus.	1	3	false	false	true	true	5	4	2	3
3	Introduction to Physics	Physics	PHYS101	3	\N	MATH201	This course provides an introduction to physics principles.	2	1	true	true	true	true	8	4	3	5
4	Distributed Databases	Computer Science	CS302	3	\N	CS101	This course provides an introduction to distributed databases.	2	2	true	true	false	true	4	2	1	5
\.


--
-- Data for Name: course_offering; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.course_offering (offering_id, course_id, semester, section_number, start_time, end_time, monday, tuesday, wednesday, thursday, friday, saturday, sunday, has_final_project, has_final_exam, textbook, class_address, allow_audit) FROM stdin;
1	1	1	1	08:00:00	10:00:00	John Smith	\N	\N	Jane Doe	\N	\N	\N	true	false	Introduction to Computer Science	123 Main St	true
2	2	1	1	10:00:00	12:00:00	\N	\N	Gilbert Strang	\N	\N	\N	\N	true	true	Advanced Calculus	456 Elm St	false
3	3	2	1	08:00:00	10:00:00	John Smith	\N	\N	Jane Doe	\N	\N	\N	false	true	Introduction to Physics	789 Oak St	true
4	4	2	1	16:00:00	18:00:00	\N	\N	John Smith	Brendan Burns	\N	\N	\N	false	true	Distributed Systems	789 Oak St	true
\.


--
-- Data for Name: course_prerequisite; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.course_prerequisite (pre_course_id, course_id) FROM stdin;
1	2
2	3
\.


--
-- Data for Name: course_tags_count; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.course_tags_count (course_id, clear_grading, pop_quiz, group_projects, inspirational, long_lectures, extra_credit, few_tests, good_feedback, tough_tests, heavy_papers, cares_for_students, heavy_assignments, respected, participation, heavy_reading, tough_grader, hilarious, would_take_again, good_lecture, no_skip) FROM stdin;
1	5	2	3	4	2	1	3	4	2	1	5	3	4	2	1	5	3	4	2	\N
2	4	1	2	3	1	2	2	3	1	2	4	2	3	1	2	4	2	3	1	\N
3	3	2	1	2	3	1	1	2	3	1	3	1	2	3	1	3	1	2	3	\N
4	2	3	0	2	3	1	1	2	3	0	3	4	2	3	5	3	1	2	3	\N
\.


--
-- Data for Name: gsi; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gsi (course_offering_id, student_id) FROM stdin;
\.


--
-- Data for Name: instructor; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.instructor (instructor_id, name, uniqname) FROM stdin;
1	John Smith	jsmith
2	Jane Doe	jdoe
3	Gilbert Strang	gstrang
4	Brendan Burns	bburns
\.


--
-- Data for Name: offering_instructor; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.offering_instructor (offering_instructor_id, offering_id, instructor_id) FROM stdin;
1	1	1
2	2	2
3	3	3
4	4	4
\.


--
-- Data for Name: program; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.program (program_id, name, college, introduction) FROM stdin;
1	Computer Science	Engineering	This program focuses on computer science principles and applications.
2	Mathematics	Arts and Sciences	This program provides a comprehensive study of mathematical concepts and theories.
3	Physics	Arts and Sciences	This program explores the fundamental principles of physics and their applications.
\.


--
-- Data for Name: program_course; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.program_course (program_id, course_id, workload, category) FROM stdin;
1	1	100	Core
1	4	80	Elective
2	2	90	Core
3	3	70	Core
\.


--
-- Data for Name: program_requirement; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.program_requirement (program_id, category, min_credit, additional_req) FROM stdin;
1	Core	120	\N
2	Core	90	\N
3	Core	200	\N
\.


--
-- Data for Name: semester; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.semester (semester_id, semester, year) FROM stdin;
1	Fall	2020
2	Spring	2021
3	Summer	2021
\.


--
-- Data for Name: student; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.student (student_id, lastname, firstname, program_id, declare_major, total_credit, total_gpa, entered_as, admit_term, predicted_graduation_semester, degree, minor, internship) FROM stdin;
1	Smith	John	1	Computer Science	120	3.5	Freshman	2018-01-01	2022-05-01	Bachelor of Science	\N	\N
2	Doe	Jane	1	Computer Science	90	3.2	Freshman	2018-01-01	2022-05-01	Bachelor of Science	\N	\N
3	Johnson	David	2	Mathematics	100	3.6	Freshman	2019-01-01	2022-05-01	Bachelor of Arts	Mathematics	\N
\.


--
-- Data for Name: student_record; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.student_record (student_id, course_id, semester, grade, how, transfer_source, earn_credit, repeat_term, test_id, offering_id) FROM stdin;
1	1	1	A	in-person	\N	Yes	\N	1	1
1	2	1	A	in-person	\N	Yes	\N	1	2
1	3	2	A	in-person	\N	Yes	\N	1	3
1	4	2	A	in-person	\N	Yes	\N	1	4
2	2	1	C	in-person	\N	Yes	\N	1	2
2	1	1	B	online	\N	Yes	\N	1	1
3	2	1	B+	in-person	\N	Yes	\N	1	2
3	4	2	B+	in-person	\N	Yes	\N	1	4
\.


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

