# -*- coding: utf-8 -*-

"""
Scrapy definition for Extra - https://www.extra.com.br/
"""

import json
import re

import scrapy
from scraper.items import ScraperItem


class Extra(scrapy.Spider):
    """ Spider definition for Extra """

    name = 'Extra'
    categories = [
        "TelefoneseCelulares/Samsung/?Filtro=C38_M459",  # Celulares
        "Eletrodomesticos/maquinadelavar/?Filtro=C13_C24",  # Máquina de Lavar
        "tv-video/Televisores/?Filtro=C1_C2",  # TVs
        "Moveis/SaladeEstar/?Filtro=C93_C97",  # Móveis Sala de Estar
        "perfumaria/Perfumes/?Filtro=C1886_C1887",  # Perfumes
        "Informatica/Notebook/?Filtro=C56_C57",  # Notebooks
    ]

    def start_requests(self):
        """ Generate initial requests """
        for category in self.categories:
            for page in range(50):
                url = self.search_url(category, page)
                yield scrapy.Request(url)

    def search_url(self, category, page):
        """ Return search URL """
        return 'https://www.extra.com.br/{}&ordenacao=_maisvendidos&paginaAtual={}'.format(category, page + 1)

    def review_url(self, product_id, page):
        """ Return review URL """
        return 'https://avaliacoes.api-extra.com.br/V1/api//produto/AvaliacoesPorProdutoPaginado?id={}&PaginaCorrente={}&QuantidadeItensPagina=5&Criterio=Data'.format( # pylint: disable=line-too-long
            product_id, page)

    def parse_reviews(self, response):
        """ Parse review page """
        data = json.loads(response.text)['avaliacao']

        if data['quantidadeAvaliacoes'] is None:
            return

        for review in data['avaliacoes']:
            yield ScraperItem(
                title=review['titulo'],
                text=review['descricao'],
                rating=review['nota'],
            )

        if data['quantidadeAvaliacoes'] > response.meta['page'] * 5:
            product_id = response.meta['id']
            next_page = response.meta['page'] + 1
            url = self.review_url(product_id, next_page)

            yield scrapy.Request(url, callback=self.parse_reviews, meta={'page': next_page, 'id': product_id})

    def parse(self, response):
        """ Parse search page """
        scripts = response.css('script[type="text/javascript"]')

        for script in scripts:
            try:
                content = re.sub(r'[\n\r]', '', script.css('::text').get())

                if "siteMetadata" in content:
                    content = re.sub(
                        r'^.*siteMetadata = (.+); *$', r'\1', content)

                    data = json.loads(content)
                    items = data['page']['listItems']

                    for item in items:
                        product_id = item['idProduct']
                        url = self.review_url(product_id, 1)
                        yield scrapy.Request(url, callback=self.parse_reviews, meta={'page': 1, 'id': product_id})
            except Exception: # pylint: disable=broad-except
                continue
