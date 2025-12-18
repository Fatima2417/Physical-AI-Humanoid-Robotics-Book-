import re
from typing import List, Dict, Tuple
from pathlib import Path

class MarkdownParser:
    def __init__(self):
        pass

    def parse_file(self, file_path: str) -> Dict:
        """Parse a markdown file and extract content with metadata"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract frontmatter if present
        frontmatter = self._extract_frontmatter(content)
        content_without_frontmatter = self._remove_frontmatter(content)

        # Extract headings and sections
        sections = self._extract_sections(content_without_frontmatter)

        # Extract title from first H1 or from filename
        title = self._extract_title(content_without_frontmatter)
        if not title:
            title = Path(file_path).stem.replace('-', ' ').replace('_', ' ').title()

        return {
            'file_path': file_path,
            'title': title,
            'frontmatter': frontmatter,
            'sections': sections,
            'raw_content': content_without_frontmatter,
            'word_count': len(content_without_frontmatter.split()),
        }

    def _extract_frontmatter(self, content: str) -> Dict:
        """Extract YAML frontmatter from markdown content"""
        frontmatter_match = re.match(r'---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter_content = frontmatter_match.group(1)
            # Simple YAML parsing (for now, just return the raw content)
            # In a full implementation, use pyyaml
            lines = frontmatter_content.split('\n')
            frontmatter = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"\'')
            return frontmatter
        return {}

    def _remove_frontmatter(self, content: str) -> str:
        """Remove frontmatter from content"""
        frontmatter_match = re.match(r'---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            return content[frontmatter_match.end():]
        return content

    def _extract_title(self, content: str) -> str:
        """Extract title from first H1 heading"""
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        return ""

    def _extract_sections(self, content: str) -> List[Dict]:
        """Extract sections based on headings"""
        sections = []

        # Split content by headings
        lines = content.split('\n')
        current_section = {
            'title': 'Introduction',  # Default title for content before first heading
            'content': '',
            'level': 0,
            'start_line': 0
        }

        line_num = 0
        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section)

                # Start new section
                hashes, title = heading_match.groups()
                current_section = {
                    'title': title.strip(),
                    'content': '',
                    'level': len(hashes),
                    'start_line': line_num
                }
            else:
                current_section['content'] += line + '\n'
            line_num += 1

        # Add the last section
        if current_section['content'].strip():
            sections.append(current_section)

        # If no sections were created (no headings), create one with all content
        if not sections and current_section['content'].strip():
            sections.append(current_section)

        return sections

    def parse_directory(self, directory_path: str, extensions: List[str] = ['.md', '.markdown']) -> List[Dict]:
        """Parse all markdown files in a directory recursively"""
        all_docs = []

        for ext in extensions:
            for file_path in Path(directory_path).rglob(f'*{ext}'):
                try:
                    doc = self.parse_file(str(file_path))
                    all_docs.append(doc)
                except Exception as e:
                    print(f"Error parsing {file_path}: {str(e)}")

        return all_docs

# Global instance
markdown_parser = MarkdownParser()