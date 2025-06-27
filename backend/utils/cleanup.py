#!/usr/bin/env python3
"""
Background cleanup services for CUDA Tutor
"""

import time
import threading
from pathlib import Path
from config import get_config

config = get_config()

class CleanupService:
    """Background service for cleaning up old files and sessions"""
    
    def __init__(self, compiler=None):
        self.compiler = compiler
        self.running = False
        self.thread = None
        self.cleanup_interval = config.CLEANUP_INTERVAL_HOURS * 3600  # Convert to seconds
        
    def start(self):
        """Start the cleanup service"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.thread.start()
        print("✅ Background cleanup service started")
    
    def stop(self):
        """Stop the cleanup service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🛑 Background cleanup service stopped")
    
    def _cleanup_loop(self):
        """Main cleanup loop"""
        while self.running:
            try:
                self._perform_cleanup()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"❌ Error in cleanup loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _perform_cleanup(self):
        """Perform all cleanup tasks"""
        print("🧹 Starting periodic cleanup...")
        
        cleaned_items = {
            'compilations': 0,
            'temp_files': 0,
            'sessions': 0
        }
        
        # Clean up old compilations
        if self.compiler and hasattr(self.compiler, 'cleanup_old_compilations'):
            try:
                cleaned_items['compilations'] = self.compiler.cleanup_old_compilations(
                    config.COMPILATION_CLEANUP_HOURS
                )
            except Exception as e:
                print(f"❌ Error cleaning compilations: {e}")
        
        # Clean up temporary files
        try:
            cleaned_items['temp_files'] = self._cleanup_temp_files()
        except Exception as e:
            print(f"❌ Error cleaning temp files: {e}")
        
        # Report cleanup results
        total_cleaned = sum(cleaned_items.values())
        if total_cleaned > 0:
            print(f"🧹 Cleanup completed: {cleaned_items}")
        else:
            print("🧹 Cleanup completed: nothing to clean")
    
    def _cleanup_temp_files(self):
        """Clean up old temporary files"""
        if not config.TEMP_DIR.exists():
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - (config.COMPILATION_CLEANUP_HOURS * 3600)
        cleaned_count = 0
        
        try:
            for item in config.TEMP_DIR.iterdir():
                if item.is_dir():
                    try:
                        # Check if directory is old
                        stat = item.stat()
                        if stat.st_mtime < cutoff_time:
                            import shutil
                            shutil.rmtree(item)
                            cleaned_count += 1
                    except Exception as e:
                        print(f"⚠️ Could not clean {item}: {e}")
        except Exception as e:
            print(f"❌ Error accessing temp directory: {e}")
        
        return cleaned_count

def start_cleanup_service(compiler=None):
    """Start the background cleanup service"""
    try:
        service = CleanupService(compiler)
        service.start()
        return service
    except Exception as e:
        print(f"❌ Failed to start cleanup service: {e}")
        return None

def cleanup_old_files(max_age_hours=1):
    """Manually clean up old files"""
    try:
        cleaned_count = 0
        
        if not config.TEMP_DIR.exists():
            return cleaned_count
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        for item in config.TEMP_DIR.iterdir():
            if item.is_dir():
                try:
                    stat = item.stat()
                    if stat.st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(item)
                        cleaned_count += 1
                        print(f"🧹 Cleaned up old directory: {item.name}")
                except Exception as e:
                    print(f"⚠️ Could not clean {item}: {e}")
        
        return cleaned_count
        
    except Exception as e:
        print(f"❌ Error in manual cleanup: {e}")
        return 0