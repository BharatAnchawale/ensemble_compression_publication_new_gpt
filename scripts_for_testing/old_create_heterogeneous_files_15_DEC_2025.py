"""
Create 15 Additional Heterogeneous Files - FIXED
Purpose: Test if ensemble works better with more mixed-content data
"""

import os
import random
import struct
import json
import base64

output_dir = "data/academic_corpora/heterogeneous_additional"
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("CREATING 15 HETEROGENEOUS FILES")
print("="*80)

# File 1: Text + Binary + JSON (500 KB)
print("\n1. Creating: mixed_text_binary_json_500kb.dat")
with open(f"{output_dir}/mixed_text_binary_json_500kb.dat", "wb") as f:
    text = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 4000
    f.write(text.encode('utf-8'))
    f.write(os.urandom(200 * 1024))
    data = {"users": [{"id": i, "name": f"User{i}", "score": random.randint(0, 1000)} 
                      for i in range(2000)]}
    f.write(json.dumps(data).encode('utf-8'))

# File 2: HTML + CSV + XML (600 KB)
print("2. Creating: mixed_html_csv_xml_600kb.dat")
with open(f"{output_dir}/mixed_html_csv_xml_600kb.dat", "wb") as f:
    html = "<html><body>" + "<p>Paragraph content here</p>" * 5000 + "</body></html>"
    f.write(html.encode('utf-8'))
    csv = "id,name,value,timestamp\n"
    csv += "\n".join([f"{i},Item{i},{random.random()},2024-12-{i%30+1:02d}" 
                      for i in range(5000)])
    f.write(csv.encode('utf-8'))
    xml = "<root>" + "".join([f"<item id='{i}'><data>{random.randint(0,1000)}</data></item>" 
                              for i in range(2000)]) + "</root>"
    f.write(xml.encode('utf-8'))

# File 3: Code + Logs + Config (700 KB)
print("3. Creating: mixed_code_logs_config_700kb.dat")
with open(f"{output_dir}/mixed_code_logs_config_700kb.dat", "wb") as f:
    code = "".join([f"def func_{i}():\n    return {i}\n" for i in range(5000)])
    f.write(code.encode('utf-8'))
    logs = "\n".join([f"[2024-12-15 10:{i%60:02d}:{i%60:02d}] INFO: Processing item {i}" 
                      for i in range(10000)])
    f.write(logs.encode('utf-8'))
    config = json.dumps({"settings": {f"key_{i}": f"value_{i}" for i in range(2000)}})
    f.write(config.encode('utf-8'))

# File 4: Compressed + Uncompressed Text (800 KB)
print("4. Creating: mixed_compressed_text_800kb.dat")
with open(f"{output_dir}/mixed_compressed_text_800kb.dat", "wb") as f:
    f.write(b"A" * 400 * 1024)
    varied = "".join([chr(65 + (i % 26)) + str(i % 10) for i in range(200000)])
    f.write(varied.encode('utf-8'))

# File 5: Binary + Structured + Random (1 MB)
print("5. Creating: mixed_binary_structured_1mb.dat")
with open(f"{output_dir}/mixed_binary_structured_1mb.dat", "wb") as f:
    for i in range(100000):
        f.write(struct.pack('i', i))
    f.write(os.urandom(400 * 1024))
    text_data = "".join([f"Sample text line number {i}\n" for i in range(10000)])
    f.write(text_data.encode('utf-8'))

# File 6: Media Headers + Data (1.2 MB)
print("6. Creating: mixed_media_simulation_1200kb.dat")
with open(f"{output_dir}/mixed_media_simulation_1200kb.dat", "wb") as f:
    header = json.dumps({"format": "video", "codec": "h264", "resolution": "1920x1080",
                        "metadata": {f"tag_{i}": f"value_{i}" for i in range(1000)}})
    f.write(header.encode('utf-8'))
    f.write(os.urandom(1000 * 1024))
    footer = json.dumps({"timestamps": [i * 0.033 for i in range(5000)]})
    f.write(footer.encode('utf-8'))

# File 7: Database-like (900 KB)
print("7. Creating: mixed_database_simulation_900kb.dat")
with open(f"{output_dir}/mixed_database_simulation_900kb.dat", "wb") as f:
    for i in range(10000):
        record = f"ID:{i:06d}|NAME:User{i}|EMAIL:user{i}@example.com|STATUS:active\n"
        f.write(record.encode('utf-8'))
    for i in range(50000):
        f.write(struct.pack('I', i))
    meta = json.dumps({"table": "users", "rows": 10000, "indexed": True})
    f.write(meta.encode('utf-8'))

# File 8: Text + Binary Interleaved (1.5 MB) - FIXED
print("8. Creating: mixed_interleaved_1500kb.dat")
with open(f"{output_dir}/mixed_interleaved_1500kb.dat", "wb") as f:
    for i in range(100):
        text_block = f"This is text block {i}\n" * 500
        f.write(text_block.encode('utf-8'))
        f.write(os.urandom(5 * 1024))

# File 9: Source Code + Executables (1 MB)
print("9. Creating: mixed_code_binary_1mb.dat")
with open(f"{output_dir}/mixed_code_binary_1mb.dat", "wb") as f:
    code = "\n".join([f"int function_{i}() {{ return {i}; }}" for i in range(10000)])
    f.write(code.encode('utf-8'))
    f.write(os.urandom(500 * 1024))

# File 10: Markdown + Images (1.3 MB)
print("10. Creating: mixed_markdown_images_1300kb.dat")
with open(f"{output_dir}/mixed_markdown_images_1300kb.dat", "wb") as f:
    md = "# Heading\n\n" + "This is a paragraph with **bold** and *italic*.\n\n" * 3000
    f.write(md.encode('utf-8'))
    fake_img = os.urandom(750 * 1024)
    b64_img = base64.b64encode(fake_img)
    f.write(b"![image](data:image/png;base64,")
    f.write(b64_img)
    f.write(b")\n")

# File 11: Config + Logs (800 KB)
print("11. Creating: mixed_config_logs_800kb.dat")
with open(f"{output_dir}/mixed_config_logs_800kb.dat", "wb") as f:
    config = "\n".join([f"[section_{i}]\nkey_{i} = value_{i}\n" for i in range(5000)])
    f.write(config.encode('utf-8'))
    logs = "\n".join([f"{i:08d} [ERROR] Exception in module {i%100}: Error code {i%1000}" 
                      for i in range(8000)])
    f.write(logs.encode('utf-8'))

# File 12: Spreadsheet-like (1.1 MB)
print("12. Creating: mixed_spreadsheet_1100kb.dat")
with open(f"{output_dir}/mixed_spreadsheet_1100kb.dat", "wb") as f:
    csv = "A,B,C,D,E,F\n"
    csv += "\n".join([",".join([str(random.random()) for _ in range(6)]) 
                      for _ in range(15000)])
    f.write(csv.encode('utf-8'))
    for i in range(125000):
        f.write(struct.pack('f', random.random()))

# File 13: Network Packets (950 KB)
print("13. Creating: mixed_network_packets_950kb.dat")
with open(f"{output_dir}/mixed_network_packets_950kb.dat", "wb") as f:
    for i in range(5000):
        header = f"PKT:{i:06d}|SRC:192.168.1.{i%255}|DST:10.0.0.{i%255}|LEN:{100+i%900:04d}|"
        f.write(header.ljust(100).encode('utf-8'))
        if i % 3 == 0:
            f.write(b"HTTP/1.1 200 OK\r\n" + os.urandom(50))
        elif i % 3 == 1:
            f.write(json.dumps({"type": "data", "seq": i}).encode('utf-8'))
        else:
            f.write(os.urandom(80))

# File 14: Backup Simulation (1.6 MB)
print("14. Creating: mixed_backup_simulation_1600kb.dat")
with open(f"{output_dir}/mixed_backup_simulation_1600kb.dat", "wb") as f:
    files_list = "\n".join([f"/path/to/file_{i}.txt|{random.randint(1000, 100000)}|2024-12-{i%30+1:02d}" 
                            for i in range(20000)])
    f.write(files_list.encode('utf-8'))
    f.write(os.urandom(800 * 1024))

# File 15: Multimedia (2 MB)
print("15. Creating: mixed_multimedia_2mb.dat")
with open(f"{output_dir}/mixed_multimedia_2mb.dat", "wb") as f:
    metadata = json.dumps({
        "tracks": [{"id": i, "codec": "aac", "bitrate": 128000, 
                   "duration": random.randint(100, 300)} for i in range(3000)]
    })
    f.write(metadata.encode('utf-8'))
    for i in range(100):
        desc = f"Chunk {i}: timestamp={i*0.1}\n" * 500
        f.write(desc.encode('utf-8'))
        f.write(os.urandom(6 * 1024))

print("\n" + "="*80)
print("✅ CREATED 15 HETEROGENEOUS FILES")
print("="*80)
print(f"Location: {output_dir}/")

files = sorted(os.listdir(output_dir))
print(f"\nFiles created: {len(files)}")
for f in files:
    size = os.path.getsize(os.path.join(output_dir, f)) / 1024
    print(f"  {f}: {size:.1f} KB")

total_size = sum([os.path.getsize(os.path.join(output_dir, f)) for f in files]) / (1024*1024)
print(f"\nTotal size: {total_size:.2f} MB")