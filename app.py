import json
from sickle import Sickle
from lxml import etree

# -----------------------------------
# CONNECT TO IIT DELHI IR (xoai format)
# -----------------------------------

sickle = Sickle(
    "https://ir.iitd.ac.in/server/oai/request"
)

# -----------------------------------
# DATE FILTER — last 5 years
# -----------------------------------

params = {
    "metadataPrefix": "xoai",
    "from": "2021-01-01",
    "until": "2026-05-14"
}

# -----------------------------------
# HELPERS: parse xoai XML
# -----------------------------------

XOAI_NS = {"x": "http://www.lyncode.com/xoai"}


def get_xoai_values(xml_element, *path_parts):
    """
    Walk the nested <element name="..."> tree in xoai XML
    and return a list of <field name="value"> texts found at the leaf.

    Example: get_xoai_values(root, "dc", "title", "none")
    """
    xpath = "/".join(
        f'x:element[@name="{p}"]' for p in path_parts
    )
    xpath += '/x:field[@name="value"]'
    nodes = xml_element.findall(xpath, XOAI_NS)
    return [n.text.strip() for n in nodes if n.text]


def get_bitstream_info(xml_element):
    """Extract PDF url from bundles/bundle[ORIGINAL]/bitstreams."""
    pdf_url = ""
    bundles = xml_element.findall(
        './/x:element[@name="bundles"]/x:element[@name="bundle"]',
        XOAI_NS
    )
    for bundle in bundles:
        name_field = bundle.find('x:field[@name="name"]', XOAI_NS)
        if name_field is not None and name_field.text == "ORIGINAL":
            url_field = bundle.find(
                './/x:element[@name="bitstream"]/x:field[@name="url"]',
                XOAI_NS
            )
            if url_field is not None and url_field.text:
                pdf_url = url_field.text.strip()
            break
    return pdf_url


def get_access_status(xml_element):
    """Extract open/restricted access status."""
    node = xml_element.find(
        './/x:element[@name="others"]/x:element[@name="access-status"]'
        '/x:field[@name="value"]',
        XOAI_NS
    )
    if node is not None and node.text:
        return node.text.strip()
    return ""


# -----------------------------------
# FETCH ALL RECORDS
# -----------------------------------

records = sickle.ListRecords(**params)

collected = []
total = 0
errors = 0

for record in records:
    total += 1

    try:
        # Deleted / header-only records have no metadata
        if not hasattr(record, "xml"):
            errors += 1
            continue

        root = record.xml
        meta = root.find(
            './/x:element[@name="dc"]',
            XOAI_NS
        )
        if meta is None:
            errors += 1
            continue

        # ── OAI identifier ──
        oai_id = ""
        hdr = root.find(
            ".//{http://www.openarchives.org/OAI/2.0/}identifier"
        )
        if hdr is not None and hdr.text:
            oai_id = hdr.text.strip()

        # ── Title ──
        title = " ".join(
            get_xoai_values(meta, "title", "none")
        )

        # ── Authors  (dc.contributor.author) ──
        authors = get_xoai_values(meta, "contributor", "author", "none")

        # ── Advisors (dc.contributor.advisor) ──
        advisors = get_xoai_values(meta, "contributor", "advisor", "none")

        # ── Subjects / Keywords ──
        subjects = get_xoai_values(meta, "subject", "none")

        # ── Dates ──
        date_issued = get_xoai_values(meta, "date", "issued", "none")
        date_created = get_xoai_values(meta, "date", "created", "none")
        date_accessioned = get_xoai_values(
            meta, "date", "accessioned", "none"
        )

        pub_date = ""
        if date_issued:
            pub_date = date_issued[0]
        elif date_created:
            pub_date = date_created[0]

        pub_year = None
        if pub_date:
            try:
                pub_year = int(pub_date[:4])
            except (ValueError, IndexError):
                pub_year = None

        # ── Type ──
        work_type = " | ".join(
            get_xoai_values(meta, "type", "none")
        )

        # ── Language ──
        lang_vals = get_xoai_values(meta, "language", "none")
        language = lang_vals[0] if lang_vals else ""

        # ── Identifier / URI ──
        uris = get_xoai_values(meta, "identifier", "uri", "none")
        landing_page_url = uris[0] if uris else ""

        # ── Relation (thesis number, series, etc.) ──
        relations = get_xoai_values(
            meta, "relation", "ispartofseries", "none"
        )
        thesis_no = relations[0] if relations else ""

        # ── Publisher ──
        publishers = get_xoai_values(meta, "publisher", "none")
        publisher = publishers[0] if publishers else ""

        # ── Description / Abstract ──
        abstract_vals = get_xoai_values(
            meta, "description", "abstract", "none"
        )
        desc_vals = get_xoai_values(meta, "description", "none")
        abstract = " ".join(abstract_vals) if abstract_vals else " ".join(desc_vals)

        # ── PDF URL from bitstreams ──
        pdf_url = get_bitstream_info(root)

        # ── Access status ──
        access_status = get_access_status(root)
        is_open_access = "open" in access_status.lower() if access_status else False

        # ── Handle ──
        handle_node = root.find(
            './/x:element[@name="others"]/x:field[@name="handle"]',
            XOAI_NS
        )
        handle = handle_node.text.strip() if (
            handle_node is not None and handle_node.text
        ) else ""

        # ── Build record dict (inspired by AI_01.json) ──
        entry = {
            "oai_id": oai_id,
            "handle": handle,
            "landing_page_url": landing_page_url,
            "pdf_url": pdf_url,
            "title": title,
            "abstract": abstract.strip() if abstract else "",
            "authors": authors,
            "advisors": advisors,
            "work_type": work_type,
            "publication_year": pub_year,
            "publication_date": pub_date,
            "language": language,
            "publisher": publisher,
            "thesis_no": thesis_no,
            "keywords": subjects,
            "is_open_access": is_open_access
        }

        collected.append(entry)

        # progress
        if total % 500 == 0:
            print(f"Processed: {total} | Collected: {len(collected)} | Errors: {errors}")

    except Exception as e:
        errors += 1
        if total % 500 == 0:
            print(f"Error at record {total}: {e}")

# -----------------------------------
# SAVE TO JSON
# -----------------------------------

output_path = "IITD_output.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(collected, f, indent=2, ensure_ascii=False)

# -----------------------------------
# RESULTS
# -----------------------------------

print("\n" + "=" * 44)
print("  IIT Delhi IR — Last 5 Years (xoai)")
print("=" * 44)
print(f"  Total Records Scanned : {total}")
print(f"  Errors / Skipped      : {errors}")
print(f"  Collected             : {len(collected)}")
print(f"  Saved to              : {output_path}")
print("=" * 44)