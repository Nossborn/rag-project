import re
import requests

from time import time, sleep
from urllib.parse import unquote
from pathlib import Path
from bs4 import BeautifulSoup

DATASET_PATH = "../dataset"
BASE_URL = "https://studieinfo.liu.se"

last_poll = 0.0
RATE_LIMIT = 1  # Minumum delay between requests, in seconds


def load_page(link: str, code: str):
    """ Loads page, either from web or disk. Caches after download. """
    save_path = Path(DATASET_PATH) / "html_files" / f"{code}.html"
    if save_path.exists():
        with open(save_path, encoding='utf-8') as f:
            return "".join(f.readlines())

    print(f"Downloading: {code}")
    # Ensure API is not hit faster than the RATE_LIMIT
    global last_poll
    if time() - last_poll < RATE_LIMIT:
        sleep(RATE_LIMIT - (time() - last_poll))

    last_poll = time()
    response = requests.get(link)
    if not response.ok:
        raise Exception("Link opening failed", response.reason)

    content = response.content.decode("utf-8")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return content


def get_courses():
    """ Return dictionary of all courses on search page. """
    content = load_page(
        f"{BASE_URL}/en/?Term=***&Type=course&MainFieldOfStudy=", "courses"
    )
    soup = BeautifulSoup(content, "html.parser")

    courses: dict[str, dict[str, str]] = {}
    pattern = re.compile(r'/en/kurs/([^/]+)')
    for course in soup.find_all("a", attrs={"class": "pseudo-h3", "href": pattern}):
        link = course["href"].lower()
        code = re.findall(pattern, link)
        code = unquote(code[0]).lower()
        name = course.text.strip().lower().replace('\n', '')
        courses[code] = {"name": name, "link": BASE_URL + link}

    return courses


def get_programs():
    """ Return dictionary of all programs. """
    content = load_page(
        f"{BASE_URL}/en/?Term=***&Type=programme&MainFieldOfStudy=", "program"
    )
    soup = BeautifulSoup(content, "html.parser")

    programs: dict[str, dict[str, str]] = {}
    pattern = re.compile(r'/program/([^/]+)')
    for program in soup.find_all("a", attrs={"class": "pseudo-h3", "href": pattern}):
        link = program["href"].lower()
        code = re.findall(pattern, link)
        code = unquote(code[0]).lower()
        name = program.text.strip().lower().replace('\n', '')
        programs[code] = {"name": name, "link": BASE_URL + link}

    return programs


def get_courses_from_program(link: str, code: str):
    """ Return dictionary of all courses in program. """
    content = load_page(link, code)
    soup = BeautifulSoup(content, "html.parser")

    courses: dict[str, dict[str, str]] = {}
    pattern = re.compile(r'/kurs/([^/]+)')
    for course in soup.find_all(attrs={'data-course-code': re.compile(r".*")}):
        anchor_tag = course.find("a")
        link = anchor_tag["href"].lower()
        code = re.findall(pattern, link)
        code = unquote(code[0]).lower()
        name = anchor_tag.text.strip().lower()
        courses[code] = {"name": name, "link": BASE_URL + link}

    return courses


def get_course_info(link: str, code: str, name: str):
    """ Return parsed course syllabus. """
    content = load_page(link, code)
    soup = BeautifulSoup(content, "html.parser")

    course_text = [
        "Course code:",
        code,
        "\nCourse title:",
        name
    ]

    def strip(s): return s.strip().replace('\n', '')

    content = soup.find(id="syllabus")

    syllabus = content.find("section", attrs={"class": "syllabus f-2col"})

    # Remove all div tags in syllabus for easier parsing
    for match in syllabus.find_all('div'):
        match.decompose()

    for child in syllabus:
        text = strip(child.text)

        if len(text) == 0:
            continue

        tag = child.name
        if tag is None:
            # Bread text
            course_text.append(text)
            continue

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            # Header
            course_text.append("\n" + text + ":")
            continue

        if tag in ("ul", "ol"):
            # List
            lis = [" - " + strip(li.text) for li in child.find_all('li')]
            course_text.extend(lis)
            continue

        if tag == "p":
            # Newline
            course_text.append("\n" + text)
            continue

        if tag == "table":
            heads = " ".join([strip(th.text) for th in child.find_all('th')])
            rows = []
            for row in child.find_all('tr'):
                rows.append(" ".join([strip(td.text)
                            for td in row.find_all('td')]))
            course_text.extend([heads, *rows])
            continue

        if tag == "a" and "http" in text:
            continue

        print("Warning! New type found:", child)

    return course_text


def save_course(course_text: list[str], code: str):
    """ Save the course after. """
    save_path = Path(DATASET_PATH) / "courses" / f"{code}.txt"

    with open(save_path, "w", encoding='utf-8') as f:
        f.write("\n".join(course_text))
