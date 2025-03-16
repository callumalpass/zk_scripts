#!/usr/bin/env python3
"""
Integration test for ZK scripts module refactoring.

This script tests the modularized functionality to ensure backward compatibility.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_bibliography_modules():
    """Test the bibliography modules."""
    try:
        # Import from new modular structure
        from zk_core.bibliography.builder import generate_citation_keys, generate_bibliography
        from zk_core.bibliography.viewer import format_bibliography_data
        
        # Check function signatures match old versions
        assert 'getbibkeys_script' in generate_citation_keys.__code__.co_varnames
        assert 'biblib_dir' in generate_citation_keys.__code__.co_varnames
        assert 'notes_dir' in generate_citation_keys.__code__.co_varnames
        
        assert 'index_file' in generate_bibliography.__code__.co_varnames
        assert 'output_paths' in generate_bibliography.__code__.co_varnames
        
        logger.info("✅ Bibliography modules import and function signatures verified")
        return True
    except ImportError as e:
        logger.error(f"❌ Bibliography module import failed: {e}")
        return False
    except AssertionError:
        logger.error("❌ Bibliography function signatures don't match expected parameters")
        return False

def test_command_module():
    """Test the commands module."""
    try:
        # Import command executor
        from zk_core.commands import CommandExecutor, run_command
        
        # Test simple command execution
        rc, stdout, stderr = CommandExecutor.run(["echo", "test"])
        assert rc == 0
        assert stdout.strip() == "test"
        
        # Test backward compatibility with run_command
        rc, stdout, stderr = run_command(["echo", "test"])
        assert rc == 0
        assert stdout.strip() == "test"
        
        logger.info("✅ Commands module functionality verified")
        return True
    except ImportError as e:
        logger.error(f"❌ Commands module import failed: {e}")
        return False
    except AssertionError:
        logger.error("❌ Commands execution didn't produce expected results")
        return False

def test_markdown_module():
    """Test the markdown module."""
    try:
        # Import markdown functions
        from zk_core.markdown import (
            extract_frontmatter_and_body,
            extract_wikilinks_filtered,
            calculate_word_count,
            extract_citations
        )
        
        # Test frontmatter extraction
        test_content = """---
title: Test Note
tags: [test, markdown]
---

# Test Heading

This is [[a link]] to another note.

See also [@citation2000].
"""
        meta, body = extract_frontmatter_and_body(test_content)
        assert isinstance(meta, dict)
        assert meta.get('title') == 'Test Note'
        assert 'This is' in body
        
        # Test wikilink extraction
        links = extract_wikilinks_filtered(body)
        assert len(links) == 1
        assert links[0] == 'a link'
        
        # Test word count
        count = calculate_word_count(body)
        assert count > 0
        
        # Test citation extraction
        citations = extract_citations(body)
        assert len(citations) == 1
        assert citations[0] == 'citation2000'
        
        logger.info("✅ Markdown module functionality verified")
        return True
    except ImportError as e:
        logger.error(f"❌ Markdown module import failed: {e}")
        return False
    except AssertionError:
        logger.error("❌ Markdown functions didn't produce expected results")
        return False

def test_legacy_imports():
    """Test that legacy imports still work for backward compatibility."""
    try:
        # Import functions from utils that should now be imported from elsewhere
        from zk_core.utils import (
            run_command,
            extract_frontmatter_and_body,
            extract_wikilinks_filtered,
            calculate_word_count,
            extract_citations,
            json_ready
        )
        
        # Make sure they're callable
        rc, stdout, stderr = run_command(["echo", "test"])
        assert rc == 0
        
        meta, body = extract_frontmatter_and_body("---\ntest: true\n---\nContent")
        assert isinstance(meta, dict)
        
        logger.info("✅ Legacy imports functionality verified")
        return True
    except ImportError as e:
        logger.error(f"❌ Legacy import failed: {e}")
        return False
    except AssertionError:
        logger.error("❌ Legacy functions didn't produce expected results")
        return False

def main():
    """Run all integration tests."""
    tests = [
        test_bibliography_modules,
        test_command_module,
        test_markdown_module,
        test_legacy_imports
    ]
    
    # Run all tests
    all_passed = True
    for test in tests:
        try:
            test_name = test.__name__
            logger.info(f"Running test: {test_name}")
            
            if not test():
                all_passed = False
                logger.error(f"Test failed: {test_name}")
            
            logger.info(f"Completed test: {test_name}\n")
        except Exception as e:
            all_passed = False
            logger.error(f"Error in test {test.__name__}: {e}")
    
    # Print final summary
    if all_passed:
        logger.info("All integration tests passed! Modules are correctly refactored.")
        return 0
    else:
        logger.error("Some integration tests failed. See error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())