// ChatsSection.js
import React, { useState, useEffect } from 'react';
import supabase from '../config/supabaseClient';
import ChatView from './ChatView';

export default function ChatsSection({ moduleId, onSendMessage }) {
  const [chats, setChats] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!moduleId) return;
    let isMounted = true;

    async function fetchChats() {
      setLoading(true);
      const { data, error } = await supabase
        .from('chats')
        .select('*')
        .eq('module_id', moduleId)
        .order('created_at', { ascending: true });

      if (!error && isMounted) {
        setChats(data);
      }
      setLoading(false);
    }

    fetchChats();

    return () => {
      isMounted = false;
    };
  }, [moduleId]);

  if (loading) return <p>Loading chats...</p>;
  if (!chats.length) return <p>No chats in this module yet.</p>;

  return (
    <ChatView chats={chats} onSendMessage={onSendMessage} />
  );
}
