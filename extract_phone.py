import pdfplumber

found = False
with pdfplumber.open('학부학칙.pdf') as pdf:
    for i in range(len(pdf.pages) - 3, len(pdf.pages)):
        page = pdf.pages[i]
        text = page.extract_text()
        if text and '전화' in text:
            print(f'Found phone directory on page {i+1}')
            found = True
            tables = page.extract_tables()
            if tables:
                with open('phone_directory.txt', 'w', encoding='utf-8') as f:
                    f.write('# 한세대학교 전화번호 안내 (전화번호부)\n\n이 내용은 한세대학교 교내 부서 전화번호부입니다.\n\n')
                    for table in tables:
                        for row in table:
                            clean_row = [str(cell).replace('\n', '') if cell else '' for cell in row]
                            if any(clean_row):
                                f.write(' | '.join(clean_row) + '\n')
                        f.write('\n')
                print('Wrote phone_directory.txt (tables)')
            else:
                print('No tables extracted by pdfplumber, falling back to text.')
                lines = text.split('\n')
                with open('phone_directory.txt', 'w', encoding='utf-8') as f:
                    f.write('# 한세대학교 전화번호 안내 (전화번호부)\n\n이 내용은 한세대학교 교내 부서 전화번호부입니다.\n\n')
                    f.write('\n'.join(lines))
            break

if not found:
    print('Phone directory not found in the last 3 pages using pdfplumber.')
    # Fallback to web search if possible
