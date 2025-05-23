import requests
#PMC11945313

def obtener_texto_completo_pmc(pmcid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    pmcid = pmcid.strip()
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml"
    }
    response = requests.get(url, params=params)
    return response.text

from xml.etree import ElementTree as ET

def extraer_secciones_pmc(xml_str):
    root = ET.fromstring(xml_str)
    secciones = []
    for sec in root.findall(".//sec"):
        titulo = sec.findtext("title", default="Sin título").strip()
        parrafos = [p.text.strip() for p in sec.findall("p") if p.text]
        texto = "\n".join(parrafos)
        secciones.append(f"== {titulo.upper()} ==\n{texto}\n")
    return "\n".join(secciones)

xml_crudo = obtener_texto_completo_pmc("PMC11945313")
contenido = extraer_secciones_pmc(xml_crudo)
print(contenido)
