from datetime import datetime

from bs4 import BeautifulSoup


class TEIFile(object):
    """
    The TEIFile class is used to read and parse a TEI (Text Encoding Initiative) file and extract information from it.
    The class takes a single parameter, the filename of the TEI file to be parsed. The class has several properties,
    such as doi, title, year, abstract, location, authors, and text. These properties are used to extract specific
    information from the TEI file and make it easily accessible. The class uses the BeautifulSoup library to parse the
    TEI file and extract the necessary information. The class also has a method _read_tei() which reads the file and
    generates a soup from the input. If the file cannot be read, the method raises a runtime error.
    """

    def __init__(self, filename):
        self.filename = filename
        self._text = None
        self._title = ''
        self._abstract = ''
        self._journal_name = ''
        self._year = None
        self._location = None

        self.soup = self._read_tei()

    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        if not idno_elem:
            return ''
        else:
            return idno_elem.getText().lower()

    @property
    def title(self):
        if not self._title:
            self._title = self.soup.title.getText()
        return self._title

    @property
    def year(self):
        if not self._year:
            if self.soup.publicationStmt.date is not None:
                date = self.soup.publicationStmt.date.text
                for fmt in ['%d %B %Y', '%B %Y']:
                    try:
                        dt = datetime.strptime(date, fmt)
                        self._year = dt.year
                    except ValueError:
                        pass
        return self._year

    @property
    def abstract(self):
        if not self._abstract:
            abstract = self.soup.abstract.getText(separator=' ', strip=True)
            self._abstract = abstract
        return self._abstract

    @property
    def location(self):
        if not self._location:
            location = self.soup.find('country')
            if location is not None:
                self._location = location.text
        return self._location

    @property
    def authors(self):
        authors_in_header = self.soup.analytic.find_all('author')

        result = []
        for author in authors_in_header:
            persname = author.persName
            if not persname:
                continue
            name = ''
            # firstname = author.find('forename',attrs={'type':'first'}).text
            if author.find('forename', attrs={'type': 'first'}) is not None:
                name = author.find('forename', attrs={'type': 'first'}).text

            if author.find('forename', attrs={'type': 'middle'}) is not None:
                name = ' '.join([name, author.find('forename', attrs={'type': 'middle'}).text])
            if author.find('surname') is not None:
                name = ' '.join([name, author.find('surname').text])
            result.append(name)
        return result

    @property
    def text(self):
        if not self._text:
            divs_text = {}
            for div in self.soup.body.find_all("div"):
                # div is neither an appendix nor references, just plain text.
                if not div.get("type"):
                    text = ''
                    for c in div.contents:
                        if c.name == 'head':
                            continue
                        text = text + ' ' + c.getText()
                    divs_text[div.next_element.text] = text

            # plain_text = " ".join(divs_text)
            self._text = divs_text
        return self._text

    def _read_tei(self):
        with open(self.filename, 'r') as tei:
            soup = BeautifulSoup(tei, 'xml')
            return soup
        raise RuntimeError('Cannot generate a soup from the input')
