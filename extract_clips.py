import xml.etree.ElementTree as ET
import json
import argparse
import sys
import os

def extract_clipitems(xml_file, output_json):
    if not os.path.exists(xml_file):
        print(f"Error: The file '{xml_file}' does not exist.")
        sys.exit(1)

    try:
        # FCP XMLs might not be perfectly well-formed, but ElementTree usually handles them
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)

    results = []
    
    # Iterate through all 'clipitem' tags in the entire XML tree
    for clipitem in root.iter('clipitem'):
        clip_id = clipitem.get('id')
        
        # Find the <start> and <end> child elements
        start_elem = clipitem.find('start')
        end_elem = clipitem.find('end')
        
        # We only add it if start and end are present, or we can add them even if missing
        start_val = start_elem.text if start_elem is not None else None
        end_val = end_elem.text if end_elem is not None else None
        
        results.append({
            'clipitem_id': clip_id,
            'start': start_val,
            'end': end_val
        })
        
    # Write the extracted data to a JSON file
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Successfully extracted {len(results)} clipitems and saved to '{output_json}'.")
    except IOError as e:
        print(f"Error writing to JSON file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clipitem id, start, and end values from FCP XML.")
    parser.add_argument("input_xml", help="Path to the input FCP XML file")
    parser.add_argument("-o", "--output", default="clipitems_output.json", help="Path to the output JSON file (default: clipitems_output.json)")
    
    args = parser.parse_args()
    extract_clipitems(args.input_xml, args.output)
