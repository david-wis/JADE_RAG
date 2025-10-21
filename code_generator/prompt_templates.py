"""
Module for managing language-specific prompt templates using Jinja2.
"""
import os
from typing import Dict, Any
from enum import Enum
from jinja2 import Environment, FileSystemLoader, Template


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    HASKELL = "haskell"


class PromptTemplates:
    """Manages language-specific prompt templates for code generation using Jinja2."""
    
    def __init__(self):
        # Get the directory where this module is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(current_dir, "templates")
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Cache for loaded templates
        self._template_cache = {}
    
    def _get_template(self, language: Language, template_name: str) -> Template:
        """Get a Jinja2 template for a specific language and template name."""
        cache_key = f"{language.value}_{template_name}"
        
        if cache_key not in self._template_cache:
            template_path = f"{language.value}/{template_name}.j2"
            try:
                template = self.jinja_env.get_template(template_path)
                self._template_cache[cache_key] = template
            except Exception as e:
                raise ValueError(f"Template '{template_path}' not found: {e}")
        
        return self._template_cache[cache_key]
    
    def get_template(self, language: Language, template_name: str) -> str:
        """Get a specific template for a given language (for backward compatibility)."""
        # This method is kept for backward compatibility but is not recommended
        # Use format_template instead for better functionality
        return self.format_template(language, template_name)
    
    def format_template(self, language: Language, template_name: str, **kwargs) -> str:
        """Get and format a template with the provided arguments."""
        template = self._get_template(language, template_name)
        return template.render(**kwargs)
    
    def get_system_message(self, language: Language, template_name: str) -> str:
        """Get the system message for a specific template and language."""
        system_template_name = f"{template_name}_system"
        return self.format_template(language, system_template_name)