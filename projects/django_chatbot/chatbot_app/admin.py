from django.contrib import admin
from .models import Conversation, Message

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'created_at', 'updated_at']
    list_filter = ['created_at']
    search_fields = ['session_id']

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'role', 'content', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['content']

