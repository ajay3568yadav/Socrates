#!/usr/bin/env python3
"""
Supabase client configuration for backend
Save as: backend/config/supabaseClient.py
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL') or os.getenv('REACT_APP_SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY') or os.getenv('REACT_APP_SUPABASE_KEY')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Optional: for server-side operations

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("❌ Supabase configuration missing!")
    print("Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
    print("You can find these in your Supabase project settings")
    raise Exception("Supabase configuration missing")

# Create Supabase client
try:
    # Use service key if available (for server-side operations), otherwise use anon key
    api_key = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_ANON_KEY
    supabase: Client = create_client(SUPABASE_URL, api_key)
    print("✅ Supabase client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Supabase client: {e}")
    raise

# Export the client
__all__ = ['supabase']