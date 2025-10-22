import re
import logging

logger = logging.getLogger(__name__)

def clean_markdown_links(text: str) -> str:
    """
    Clean markdown links by keeping only the title text and removing the URL.
    
    Examples:
    - "[Google](https://google.com)" -> "Google"
    - "[Python Docs](https://docs.python.org)" -> "Python Docs"
    - "Text with [link](url) in middle" -> "Text with link in middle"
    
    Args:
        text: The text containing markdown links
        
    Returns:
        Text with markdown links cleaned (only title kept)
    """
    # Pattern to match markdown links: [title](url)
    # This handles both simple and complex URLs
    markdown_link_pattern = r'\[([^\]]+)\]\([^)]+\)'
    
    # Replace markdown links with just the title text
    cleaned_text = re.sub(markdown_link_pattern, r'\1', text)
    
    return cleaned_text

def clean_html_images(text: str) -> str:
    """
    Remove HTML img tags from text.
    
    Examples:
    - '<img src="image.jpg" alt="description">' -> ''
    - 'Text <img src="pic.png"> more text' -> 'Text  more text'
    - '<img src="test.jpg" width="100" height="50">' -> ''
    
    Args:
        text: The text containing HTML img tags
        
    Returns:
        Text with HTML img tags removed
    """
    # Pattern to match HTML img tags with any attributes
    # This handles img tags with various attributes like src, alt, width, height, etc.
    html_img_pattern = r'<img[^>]*>'
    
    # Remove all img tags
    cleaned_text = re.sub(html_img_pattern, '', text)
    
    return cleaned_text

def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags but preserve their content.
    
    Examples:
    - '<p>Hello world</p>' -> 'Hello world'
    - '<strong>Bold text</strong>' -> 'Bold text'
    - '<div class="container">Content</div>' -> 'Content'
    - '<kbd>Ctrl</kbd>+<kbd>Enter</kbd>' -> 'Ctrl+Enter'
    
    Args:
        text: The text containing HTML tags
        
    Returns:
        Text with HTML tags removed but content preserved
    """
    # Pattern to match HTML tags (opening and closing tags)
    # This handles both self-closing tags like <br/> and paired tags like <p></p>
    html_tag_pattern = r'<[^>]+>'
    
    # Remove all HTML tags
    cleaned_text = re.sub(html_tag_pattern, '', text)
    
    return cleaned_text

def normalize_headers(text: str) -> str:
    """
    Normalize header formatting from Haskell-style to markdown format.
    
    Converts:
    - Text followed by ====== to ## Text
    - Text followed by ------ to ### Text
    
    Only processes headers outside of code blocks.
    
    Examples:
    - "Title\n=====" -> "## Title"
    - "Subtitle\n------" -> "### Subtitle"
    - "Code\n=====\n```haskell\ncode```" -> "## Code\n```haskell\ncode```"
    
    Args:
        text: The text containing Haskell-style headers
        
    Returns:
        Text with normalized markdown headers
    """
    lines = text.split('\n')
    normalized_lines = []
    i = 0
    in_code_block = False
    
    while i < len(lines):
        line = lines[i]
        
        # Track code block state
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
        
        # Only process headers outside of code blocks
        if not in_code_block and i + 1 < len(lines):
            next_line = lines[i + 1]
            
            # Check for ====== pattern (main headers)
            if re.match(r'^=+$', next_line.strip()):
                # Convert to ## header
                normalized_lines.append(f"## {line.strip()}")
                i += 2  # Skip the underline line
                continue
            
            # Check for ------ pattern (sub headers)
            elif re.match(r'^-+$', next_line.strip()):
                # Convert to ### header
                normalized_lines.append(f"### {line.strip()}")
                i += 2  # Skip the underline line
                continue
        
        # If no header pattern or inside code block, keep the line as is
        normalized_lines.append(line)
        i += 1
    
    return '\n'.join(normalized_lines)

def preprocess_notebook_content(text: str) -> str:
    """
    Apply all preprocessing steps to notebook content.
    
    This function:
    1. Normalizes headers (===== and ----- to ## and ###)
    2. Cleans markdown links (keeps only title text)
    3. Removes HTML img tags
    4. Removes all HTML tags (preserving content)
    5. Cleans up extra whitespace
    
    Args:
        text: The raw notebook content
        
    Returns:
        Preprocessed text with headers normalized, links and HTML cleaned
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text provided for preprocessing")
        return text or ""
    
    # Apply preprocessing steps
    cleaned_text = text
    
    # Step 1: Normalize headers (do this first to preserve line structure)
    cleaned_text = normalize_headers(cleaned_text)
    
    # Step 2: Clean markdown links
    cleaned_text = clean_markdown_links(cleaned_text)
    
    # Step 3: Remove HTML images
    cleaned_text = clean_html_images(cleaned_text)
    
    # Step 4: Remove all HTML tags (preserving content)
    cleaned_text = clean_html_tags(cleaned_text)
    
    # Step 5: Clean up extra whitespace (preserve line breaks)
    # Replace multiple consecutive spaces/tabs with single space, but preserve newlines
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    
    # Remove leading/trailing whitespace from each line
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

def preprocess_cell_content(cell_text: str) -> str:
    """
    Preprocess individual cell content.
    
    This is a wrapper around preprocess_notebook_content for consistency
    and potential future cell-specific preprocessing.
    
    Args:
        cell_text: The content of a single notebook cell
        
    Returns:
        Preprocessed cell content
    """
    return preprocess_notebook_content(cell_text)
