import streamlit as st
import os
import tempfile
import base64
from urllib.parse import urlparse
import requests
import io
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD
from pyvis.network import Network
from google import genai
from google.genai import types
import asyncio
import re

def normalize_string(input_str):
    # Trim extra spaces and replace internal spaces with underscores
    cleaned = re.sub(r'\s+', ' ', input_str.strip())
    cleaned = cleaned.replace(' ', '_')
    
    # Remove all special characters (keep only alphanumeric and underscores)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', cleaned)
    
    # Take first 100 characters
    return cleaned[:100]

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Configure page settings
st.set_page_config(
    page_title="Multimodal Knowledge Graph Creator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and introduction
st.title("Multimodal Knowledge Graph Creator")
st.markdown("""
This app creates knowledge graphs from multimodal inputs (text, images, audio, video, PDFs) using Google's Gemini.
Upload files or provide URLs, and Gemini will extract entities and relationships to build an interactive knowledge graph.
""")

# Initialize session state variables if they don't exist
if 'kg_graph' not in st.session_state:
    st.session_state.kg_graph = rdflib.Graph()
    # Define namespaces
    st.session_state.ns = Namespace("http://example.org/kg/")
    st.session_state.kg_graph.bind("kg", st.session_state.ns)
    st.session_state.kg_graph.bind("rdf", RDF)
    st.session_state.kg_graph.bind("rdfs", RDFS)
    
    # Define media-related properties in the namespace
    st.session_state.kg_graph.add((st.session_state.ns.mediaType, RDF.type, RDF.Property))
    st.session_state.kg_graph.add((st.session_state.ns.mediaType, RDFS.label, Literal("Media Type")))
    
    st.session_state.kg_graph.add((st.session_state.ns.mediaUrl, RDF.type, RDF.Property))
    st.session_state.kg_graph.add((st.session_state.ns.mediaUrl, RDFS.label, Literal("Media URL")))
    
    st.session_state.kg_graph.add((st.session_state.ns.mediaFilename, RDF.type, RDF.Property))
    st.session_state.kg_graph.add((st.session_state.ns.mediaFilename, RDFS.label, Literal("Media Filename")))
    
    st.session_state.kg_graph.add((st.session_state.ns.mediaContent, RDF.type, RDF.Property))
    st.session_state.kg_graph.add((st.session_state.ns.mediaContent, RDFS.label, Literal("Media Content")))
    
    st.session_state.kg_graph.add((st.session_state.ns.mediaBase64, RDF.type, RDF.Property))
    st.session_state.kg_graph.add((st.session_state.ns.mediaBase64, RDFS.label, Literal("Media Base64")))
    
    st.session_state.kg_graph.add((st.session_state.ns.mediaMimeType, RDF.type, RDF.Property))
    st.session_state.kg_graph.add((st.session_state.ns.mediaMimeType, RDFS.label, Literal("Media MIME Type")))
    
    st.session_state.kg_graph.add((st.session_state.ns.mediaRef, RDF.type, RDF.Property))
    st.session_state.kg_graph.add((st.session_state.ns.mediaRef, RDFS.label, Literal("Media Reference ID")))

if 'last_processed' not in st.session_state:
    st.session_state.last_processed = None

if 'api_initialized' not in st.session_state:
    st.session_state.api_initialized = False
    st.session_state.client = None

if 'entities' not in st.session_state:
    st.session_state.entities = []

if 'relationships' not in st.session_state:
    st.session_state.relationships = []

if 'visualization_html' not in st.session_state:
    st.session_state.visualization_html = None

if 'media_files' not in st.session_state:
    st.session_state.media_files = {}

# Update the session state to include media files dictionary
if 'media_files_data' not in st.session_state:
    st.session_state.media_files_data = {}

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.text_input("Google API Key", type="password", 
                          help="Enter your Google API key for Gemini access")
    
    # Model selection
    model = st.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
        index=0,
        help="Select which Gemini model to use"
    )
    
    # Temperature setting
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        help="Lower values make output more deterministic, higher values more creative"
    )
    
    # Max output tokens
    max_tokens = st.slider(
        "Max Output Tokens", 
        min_value=1024, 
        max_value=8192, 
        value=4096, 
        step=512,
        help="Maximum number of tokens in the response"
    )
    
    # Initialize API button
    if st.button("Initialize API"):
        if api_key:
            try:
                # Initialize the client directly with the API key
                st.session_state.client = genai.Client(api_key=api_key)
                st.session_state.model_name = model
                st.session_state.generation_config = types.GenerateContentConfig(
                    response_mime_type='application/json',
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
                st.session_state.api_initialized = True
                st.success("✅ API initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing API: {str(e)}")
        else:
            st.warning("⚠️ Please enter an API key")
    
    # Clear knowledge graph button
    if st.button("Clear Knowledge Graph"):
        st.session_state.kg_graph = rdflib.Graph()
        # Reinitialize namespaces
        st.session_state.ns = Namespace("http://example.org/kg/")
        st.session_state.kg_graph.bind("kg", st.session_state.ns)
        st.session_state.kg_graph.bind("rdf", RDF)
        st.session_state.kg_graph.bind("rdfs", RDFS)
        st.session_state.entities = []
        st.session_state.relationships = []
        st.session_state.visualization_html = None
        st.success("✅ Knowledge graph cleared!")

# Function to download file from URL
def download_from_url(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    # Get file extension from URL
    parsed_url = urlparse(url)
    path = parsed_url.path
    file_extension = os.path.splitext(path)[1].lower()
    
    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        temp_file.write(response.content)
        return temp_file.name

# Function to get MIME type from file extension
def get_mime_type(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain'
    }
    
    return mime_types.get(extension, 'application/octet-stream')

# Function to process media with Gemini
# Function to process media with Gemini - modified to include explicit media references
def process_with_gemini(content_parts, media_references=None, user_annotation=None):
    """
    Process content with Gemini, providing explicit media references.
    
    Args:
        content_parts: List of content parts for Gemini
        media_references: Dictionary mapping content part index to media reference ID
        user_annotation: Optional user annotation text
    """
    if not st.session_state.api_initialized:
        st.error("⚠️ Please initialize the API first")
        return None
    
    try:
        # Get existing entities for context
        existing_entities = []
        for entity in st.session_state.entities:
            existing_entities.append({
                "id": entity.get("id", ""),
                "name": entity.get("name", ""),
                "type": entity.get("type", ""),
                "description": entity.get("description", "")
            })

        # Create context about existing knowledge graph
        existing_kg_context = ""
        if existing_entities:
            existing_kg_context = f"""
            EXISTING KNOWLEDGE GRAPH CONTEXT:
            The knowledge graph already contains {len(existing_entities)} entities. 
            Here are the key entities that you should consider connecting to:
            
            {json.dumps(existing_entities, indent=2)}
            
            IMPORTANT: When analyzing new content, try to identify connections to these existing entities.
            """
        
        # Add user annotation if provided
        annotation_text = ""
        if user_annotation:
            annotation_text = f"""
            USER ANNOTATION:
            {user_annotation}
            
            This annotation provides additional context for understanding the media content.
            """
        
        # Create media references section if provided
        media_refs_text = ""
        if media_references and len(media_references) > 0:
            media_refs_text = """
            MEDIA REFERENCES:
            The following media files are being analyzed. You MUST use these exact reference IDs
            when creating entities for media content:
            
            """
            print("media_references", media_references.items())
            for idx, (content_idx, ref_id) in enumerate(media_references.items()):
                content_type = "unknown"
                
                if content_idx < len(content_parts):
                    if hasattr(content_parts[content_idx], "mime_type"):
                        content_type = content_parts[content_idx].mime_type
                    elif hasattr(content_parts[content_idx], "text"):
                        content_type = "text"

                if "img" in ref_id:
                    content_type = "image"
                
                media_refs_text += f"{idx+1}. Content item {content_idx}: Reference ID = \"{ref_id}\", Type = {content_type}\n"
                print(media_refs_text, content_parts)
            
            media_refs_text += """
            IMPORTANT: When you identify entities that represent media content (images, audio, video files), 
            you MUST use these exact reference IDs in the "media_ref" field of the entity. Do not make up
            your own reference IDs.
            """
            
        # Add KG extraction prompt
        prompt = f"""
        You are a multimodal knowledge extraction assistant specialized in creating knowledge graphs.
        
        {existing_kg_context}
        
        {annotation_text}
        
        {media_refs_text}
        
        Analyze the provided content (which may include text, images, audio, video, or PDFs) and extract:
        1. All important entities (people, organizations, concepts, objects, locations, etc.)
        2. The meaningful relationships between these entities
        3. For visual media (images/videos), identify specific visual elements and their properties
        4. For audio, identify speakers, topics, music, sounds, etc.
        
        Be specific and detailed - avoid generic entity names like just "person" or "object".
        Extract proper names, specific identifiers, and unique characteristics.
        
        For media content:
        - Create entities for the media files themselves using the exact reference IDs provided
        - Create entities for distinctive elements in the media (faces, objects, text, scenes)
        - Include visual/audio descriptors
        
        Format your response as a JSON object with two keys:
        - "entities": an array of entity objects, each with:
          - "id": a unique identifier for the entity (use snake_case)
          - "name": the display name of the entity
          - "type": the entity type (PERSON, ORGANIZATION, CONCEPT, etc.)
          - "description": a brief description of the entity
          - "media_type": for media entities, indicate "image", "audio", "video", etc.
          - "media_ref": for media entities, the EXACT reference ID from the provided list
          
        - "relationships": an array of relationship objects, each with:
          - "source": the id of the source entity
          - "target": the id of the target entity
          - "type": the relationship type (works_for, located_in, part_of, etc.)
          - "description": optional details about the relationship
        
        Ensure your output is valid JSON. Only include entities and relationships that are clearly present or strongly implied in the content.
        """
        prompt_part = types.Part.from_text(text=prompt)
        
        # Combine prompt with content parts
        full_content = [prompt_part] + content_parts
        
        # Call Gemini API using the client
        response = st.session_state.client.models.generate_content(
            model=st.session_state.model_name,
            contents=full_content,
            config=st.session_state.generation_config
        )
        
        # Parse JSON response
        try:
            kg_data = json.loads(response.text)
            return kg_data
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            response_text = response.text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    json_text = response_text[json_start:json_end]
                    kg_data = json.loads(json_text)
                    return kg_data
                except json.JSONDecodeError:
                    st.error("❌ Failed to parse Gemini response as JSON")
                    st.code(response_text, language="json")
                    return None
            else:
                st.error("❌ Failed to extract JSON from Gemini response")
                st.code(response_text, language="text")
                return None
            
    except Exception as e:
        st.error(f"❌ Error calling Gemini API: {str(e)}")
        return None


# Function to add entities and relationships to the knowledge graph
def add_to_knowledge_graph(kg_data, media_files=None):
    if not kg_data:
        return 0, 0
    
    # Initialize entity count
    new_entities = 0
    new_relationships = 0
    
    # Process entities
    entities = kg_data.get("entities", [])
    for entity in entities:
        # Skip if entity has no ID or name
        if not entity.get("id") and not entity.get("name"):
            continue
            
        # Get or create entity ID
        entity_id = entity.get("id", "")
        if not entity_id:
            entity_id = clean_uri_string(entity.get("name", "entity"))
        
        # Create a URI for the entity
        entity_uri = st.session_state.ns[entity_id]
        
        # Add entity type
        entity_type = entity.get("type", "CONCEPT").upper()
        type_uri = st.session_state.ns[entity_type.capitalize()]
        st.session_state.kg_graph.add((type_uri, RDF.type, RDFS.Class))
        st.session_state.kg_graph.add((entity_uri, RDF.type, type_uri))
        
        # Add entity name
        if "name" in entity:
            st.session_state.kg_graph.add((entity_uri, RDFS.label, Literal(entity["name"])))
        
        # Add entity description
        if "description" in entity:
            st.session_state.kg_graph.add((entity_uri, RDFS.comment, Literal(entity["description"])))
        
        # Add media type and reference if present
        if "media_type" in entity:
            st.session_state.kg_graph.add((entity_uri, st.session_state.ns.mediaType, Literal(entity["media_type"])))
        
        if "media_ref" in entity and media_files:
            media_ref = entity["media_ref"]
            if media_ref in media_files:
                # Add the media file reference
                media_file = media_files[media_ref]
                
                # Store media content info
                if isinstance(media_file, dict):
                    # For URLs or other references
                    if "url" in media_file:
                        st.session_state.kg_graph.add((entity_uri, st.session_state.ns.mediaUrl, Literal(media_file["url"])))
                    if "mime_type" in media_file:
                        st.session_state.kg_graph.add((entity_uri, st.session_state.ns.mediaMimeType, Literal(media_file["mime_type"])))
                    if "content" in media_file and media_file["content"]:
                        # Store base64 encoded content
                        content_node = BNode()
                        st.session_state.kg_graph.add((entity_uri, st.session_state.ns.mediaContent, content_node))
                        st.session_state.kg_graph.add((content_node, st.session_state.ns.mediaBase64, Literal(media_file["content"])))
                    if "filename" in media_file:
                        st.session_state.kg_graph.add((entity_uri, st.session_state.ns.mediaFilename, Literal(media_file["filename"])))
                else:
                    # For uploaded files that are file objects
                    try:
                        st.session_state.kg_graph.add((entity_uri, st.session_state.ns.mediaFilename, Literal(media_file.name)))
                    except:
                        # Handle the case where media_file might not have a name attribute
                        pass
                
                # Also store the reference ID for easier lookup
                st.session_state.kg_graph.add((entity_uri, st.session_state.ns.mediaRef, Literal(media_ref)))
        
        # Add entity to session state if not already there
        if not any(e.get("id") == entity_id for e in st.session_state.entities):
            st.session_state.entities.append(entity)
            new_entities += 1
    
    # Process relationships
    relationships = kg_data.get("relationships", [])
    for relationship in relationships:
        # Skip if relationship has no source or target
        if not relationship.get("source") or not relationship.get("target"):
            continue
            
        # Get source and target IDs
        source_id = relationship.get("source", "")
        target_id = relationship.get("target", "")
        
        # Create URIs for the source and target
        source_uri = st.session_state.ns[source_id]
        target_uri = st.session_state.ns[target_id]
        
        # Add relationship type
        rel_type = relationship.get("type", "related_to")
        rel_uri = st.session_state.ns[clean_uri_string(rel_type)]
        
        # Define the relationship as a property
        st.session_state.kg_graph.add((rel_uri, RDF.type, RDF.Property))
        st.session_state.kg_graph.add((rel_uri, RDFS.label, Literal(rel_type)))
        
        # Add the relationship
        st.session_state.kg_graph.add((source_uri, rel_uri, target_uri))
        
        # Add relationship description if available
        if "description" in relationship:
            # Create a reified statement for the description
            stmt = BNode()
            st.session_state.kg_graph.add((stmt, RDF.type, RDF.Statement))
            st.session_state.kg_graph.add((stmt, RDF.subject, source_uri))
            st.session_state.kg_graph.add((stmt, RDF.predicate, rel_uri))
            st.session_state.kg_graph.add((stmt, RDF.object, target_uri))
            st.session_state.kg_graph.add((stmt, RDFS.comment, Literal(relationship["description"])))
        
        # Add relationship to session state if not already there
        rel_key = f"{source_id}:{rel_type}:{target_id}"
        if not any(f"{r.get('source')}:{r.get('type')}:{r.get('target')}" == rel_key for r in st.session_state.relationships):
            st.session_state.relationships.append(relationship)
            new_relationships += 1
    
    return new_entities, new_relationships

# Function to clean strings for URIs
def clean_uri_string(text):
    if not text:
        return "item"
    
    # Replace spaces and special characters
    clean = text.replace(' ', '_')
    clean = ''.join(c for c in clean if c.isalnum() or c == '_')
    
    # Ensure it starts with a letter or underscore
    if clean and not (clean[0].isalpha() or clean[0] == '_'):
        clean = 'x_' + clean
        
    return clean.lower()  # Use lowercase for consistency

# Function to convert RDF graph to NetworkX for visualization
def graph_to_networkx():
    G = nx.DiGraph()
    
    # Get all entities
    for s, p, o in st.session_state.kg_graph:
        if p == RDF.type and o != RDF.Property and not str(o).startswith(str(RDF._NS)):
            entity_id = str(s).split('/')[-1]
            entity_type = str(o).split('/')[-1]
            
            # Get entity name from label
            entity_name = entity_id
            for _, _, name in st.session_state.kg_graph.triples((s, RDFS.label, None)):
                entity_name = str(name)
                break
            
            # Get entity description
            description = ""
            for _, _, desc in st.session_state.kg_graph.triples((s, RDFS.comment, None)):
                description = str(desc)
                break
            
            # Get media properties
            media_type = None
            media_url = None
            media_filename = None
            media_base64 = None
            
            for _, _, mt in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaType, None)):
                media_type = str(mt)
            
            for _, _, url in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaUrl, None)):
                media_url = str(url)
            
            for _, _, filename in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaFilename, None)):
                media_filename = str(filename)
            
            # Get base64 content if available
            for _, content_node, _ in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaContent, None)):
                for _, _, base64_content in st.session_state.kg_graph.triples((content_node, st.session_state.ns.mediaBase64, None)):
                    media_base64 = str(base64_content)
            
            # Add node with attributes
            G.add_node(
                entity_id, 
                name=entity_name, 
                type=entity_type,
                description=description,
                media_type=media_type,
                media_url=media_url,
                media_filename=media_filename,
                media_base64=media_base64
            )
    
    # Get all relationships
    for s, p, o in st.session_state.kg_graph:
        if p != RDF.type and p != RDFS.label and p != RDFS.comment:
            if not isinstance(o, Literal) and not str(p).startswith(str(RDF._NS)):
                source_id = str(s).split('/')[-1]
                target_id = str(o).split('/')[-1]
                rel_type = str(p).split('/')[-1]
                
                # Only add edge if both nodes exist
                if source_id in G and target_id in G:
                    # Get relationship description if available
                    description = ""
                    for stmt in st.session_state.kg_graph.subjects(RDF.subject, s):
                        if (stmt, RDF.predicate, p) in st.session_state.kg_graph and (stmt, RDF.object, o) in st.session_state.kg_graph:
                            for _, _, desc in st.session_state.kg_graph.triples((stmt, RDFS.comment, None)):
                                description = str(desc)
                                break
                    
                    G.add_edge(source_id, target_id, type=rel_type, description=description)
    
    return G

# Function to create visualization using PyVis
# Update the graph_to_networkx function to properly handle session state data

def graph_to_networkx():
    G = nx.DiGraph()
    
    # Get all entities
    for s, p, o in st.session_state.kg_graph:
        if p == RDF.type and o != RDF.Property and not str(o).startswith(str(RDF._NS)):
            entity_id = str(s).split('/')[-1]
            entity_type = str(o).split('/')[-1]
            
            # Get entity name from label
            entity_name = entity_id
            for _, _, name in st.session_state.kg_graph.triples((s, RDFS.label, None)):
                entity_name = str(name)
                break
            
            # Get entity description
            description = ""
            for _, _, desc in st.session_state.kg_graph.triples((s, RDFS.comment, None)):
                description = str(desc)
                break
            
            # Get media properties
            media_type = None
            media_url = None
            media_filename = None
            media_base64 = None
            media_ref = None
            
            for _, _, mt in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaType, None)):
                media_type = str(mt)
            
            for _, _, url in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaUrl, None)):
                media_url = str(url)
            
            for _, _, filename in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaFilename, None)):
                media_filename = str(filename)
            
            # Get media reference ID
            for _, _, ref in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaRef, None)):
                media_ref = str(ref)
                # If we have this reference in session state, get the content
                if media_ref in st.session_state.media_files_data:
                    if st.session_state.media_files_data[media_ref].get("content"):
                        media_base64 = st.session_state.media_files_data[media_ref]["content"]
                
            # Get base64 content if available directly from graph
            if not media_base64:
                for _, content_node, _ in st.session_state.kg_graph.triples((s, st.session_state.ns.mediaContent, None)):
                    for _, _, base64_content in st.session_state.kg_graph.triples((content_node, st.session_state.ns.mediaBase64, None)):
                        media_base64 = str(base64_content)
            
            # Add node with attributes
            G.add_node(
                entity_id, 
                name=entity_name, 
                type=entity_type,
                description=description,
                media_type=media_type,
                media_url=media_url,
                media_filename=media_filename,
                media_base64=media_base64,
                media_ref=media_ref
            )
    
    # Get all relationships
    for s, p, o in st.session_state.kg_graph:
        if p != RDF.type and p != RDFS.label and p != RDFS.comment:
            if not isinstance(o, Literal) and not str(p).startswith(str(RDF._NS)):
                source_id = str(s).split('/')[-1]
                target_id = str(o).split('/')[-1]
                rel_type = str(p).split('/')[-1]
                
                # Only add edge if both nodes exist
                if source_id in G and target_id in G:
                    # Get relationship description if available
                    description = ""
                    for stmt in st.session_state.kg_graph.subjects(RDF.subject, s):
                        if (stmt, RDF.predicate, p) in st.session_state.kg_graph and (stmt, RDF.object, o) in st.session_state.kg_graph:
                            for _, _, desc in st.session_state.kg_graph.triples((stmt, RDFS.comment, None)):
                                description = str(desc)
                                break
                    
                    G.add_edge(source_id, target_id, type=rel_type, description=description)
    
    return G

# Update the create_visualization function to improve image rendering
def create_visualization():
    # Convert to NetworkX graph
    nx_graph = graph_to_networkx()
    
    # Handle empty graph
    if len(nx_graph.nodes) == 0:
        return "<p>No entities in the knowledge graph yet.</p>"
    
    # Create PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=True)
    
    # Set physics options for better layout
    net.barnes_hut(
        gravity=-1200,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
        overlap=0
    )
    
    # Get entity types for coloring
    entity_types = {}
    for node, data in nx_graph.nodes(data=True):
        entity_types[data.get('type', 'Unknown')] = True
    
    # Generate colors for entity types
    colors = {}
    color_palette = [
        "#FF5733", "#33FF57", "#3357FF", "#FF33F5", 
        "#F5FF33", "#33FFF5", "#F533FF", "#5733FF"
    ]
    
    for i, entity_type in enumerate(entity_types.keys()):
        colors[entity_type] = color_palette[i % len(color_palette)]
    
    # Add nodes to the network
    for node_id, node_data in nx_graph.nodes(data=True):
        node_type = node_data.get('type', 'Unknown')
        node_name = node_data.get('name', node_id)
        
        # Check if this is a media entity
        media_type = node_data.get("media_type")
        
        # Create simple text tooltip (HTML doesn't render correctly)
        tooltip = f"Type: {node_type}"
        if node_data.get("description"):
            tooltip += f"\nDescription: {node_data['description']}"
        
        # Check specifically for image data
        is_image_node = False
        image_data_url = None
        
        if media_type == "image":
            # Check all sources of image data
            if node_data.get("media_base64"):
                # Using your approach from the working example
                image_data_url = "data:image/jpeg;base64," + node_data["media_base64"]
                is_image_node = True
            elif node_data.get("media_ref") and node_data["media_ref"] in st.session_state.media_files_data:
                media_ref = node_data["media_ref"]
                if st.session_state.media_files_data[media_ref].get("content"):
                    image_data_url = "data:image/jpeg;base64," + st.session_state.media_files_data[media_ref]["content"]
                    is_image_node = True
            print("Image", node_name, media_type, node_data.get("media_ref"), is_image_node)
        
        # Add node with appropriate style
        if is_image_node and image_data_url:
            # Using "image" shape as in your working example
            net.add_node(
                node_id,
                label=node_name,
                title=tooltip,
                shape="image",  # Using "image" instead of "circularImage"
                image=image_data_url,
                size=40
            )
        elif media_type in ["audio", "video"]:
            # Special handling for audio/video
            net.add_node(
                node_id,
                label=node_name,
                title=tooltip,
                color=colors.get(node_type, "#DDDDDD"),
                shape="icon",
                icon=dict(face="FontAwesome", code=media_type, size=25)
            )
        else:
            # Standard nodes for other entities
            net.add_node(
                node_id,
                label=node_name,
                title=tooltip,
                color=colors.get(node_type, "#DDDDDD"),
                shape="dot",
                size=25
            )
    
    # Add edges to the network
    for source, target, edge_data in nx_graph.edges(data=True):
        rel_type = edge_data.get('type', 'related_to')
        
        # Add edge with label
        net.add_edge(
            source,
            target,
            title=rel_type,  # Simple text tooltip
            label=rel_type,
            arrows="to"
        )
    
    # Enable physics and other options
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 14
        }
      },
      "edges": {
        "font": {
          "size": 12
        },
        "smooth": {
          "type": "continuous"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -1200,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.05
        },
        "minVelocity": 0.75
      }
    }
    """)
    
    # Generate HTML
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html = f.read()
            os.unlink(tmp.name)
        return html
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Function to create visualization for a filtered graph
def create_filtered_visualization(nx_graph):
    # Handle empty graph
    if len(nx_graph.nodes) == 0:
        return "<p>No entities in the filtered graph.</p>"
    
    # Create PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=True)
    
    # Set physics options for better layout
    net.barnes_hut(
        gravity=-1200,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
        overlap=0
    )
    
    # Get entity types for coloring
    entity_types = {}
    for node, data in nx_graph.nodes(data=True):
        entity_types[data.get('type', 'Unknown')] = True
    
    # Generate colors for entity types
    colors = {}
    color_palette = [
        "#FF5733", "#33FF57", "#3357FF", "#FF33F5", 
        "#F5FF33", "#33FFF5", "#F533FF", "#5733FF"
    ]
    
    for i, entity_type in enumerate(entity_types.keys()):
        colors[entity_type] = color_palette[i % len(color_palette)]
    
    # Add nodes to the network
    for node_id, node_data in nx_graph.nodes(data=True):
        node_type = node_data.get('type', 'Unknown')
        node_name = node_data.get('name', node_id)
        
        # Add node with color based on type
        net.add_node(
            node_id,
            label=node_name,
            title=f"Type: {node_type}",
            color=colors.get(node_type, "#DDDDDD"),
            size=25
        )
    
    # Add edges to the network
    for source, target, edge_data in nx_graph.edges(data=True):
        rel_type = edge_data.get('type', 'related_to')
        
        # Get weight
        weight = edge_data.get('weight', 1.0)
        
        # Adjust width based on weight
        width = 1 + (weight * 5)  # Scale to 1-6 width
        
        # Add edge with label and width based on weight
        net.add_edge(
            source,
            target,
            title=f"{rel_type} (strength: {weight:.2f})",
            label=rel_type,
            width=width,
            arrows="to"
        )
    
    # Generate HTML
    try:
        # Save to a temporary file and read it back
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html = f.read()
            os.unlink(tmp.name)
        return html
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Function to create a static image of the graph for download
def create_graph_image():
    # Convert to NetworkX graph
    G = graph_to_networkx()
    
    # Handle empty graph
    if len(G.nodes) == 0:
        return None
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Generate a layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node colors based on type
    node_types = [data.get('type', 'Unknown') for _, data in G.nodes(data=True)]
    unique_types = list(set(node_types))
    color_map = {t: plt.cm.tab10(i % 10) for i, t in enumerate(unique_types)}
    node_colors = [color_map[t] for t in node_types]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=700,
        alpha=0.8
    )
    
    # Draw node labels
    labels = {node: data.get('name', node) for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_family="sans-serif"
    )
    
    # Get edge weights for width scaling
    edge_weights = [data.get('weight', 1.0) * 2 for _, _, data in G.edges(data=True)]
    
    # Draw edges with width based on weight
    nx.draw_networkx_edges(
        G, pos,
        width=edge_weights,
        alpha=0.7,
        arrows=True
    )
    
    # Draw edge labels
    edge_labels = {(s, t): data.get('type', '') for s, t, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8
    )
    
    # Add legend for node types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color_map[t], markersize=10, label=t)
                      for t in unique_types]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and turn off axis
    plt.title("Knowledge Graph")
    plt.axis("off")
    
    # Save to BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    
    return buf

# Function to get download link for the graph image
def get_image_download_link(img_buf, filename, text):
    if img_buf is None:
        return ""
    
    img_str = base64.b64encode(img_buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to get download link for the graph in RDF format
def get_rdf_download_link(format='turtle'):
    try:
        # Serialize the graph to the specified format
        if format == 'turtle':
            content_type = 'text/turtle'
            extension = 'ttl'
            data = st.session_state.kg_graph.serialize(format='turtle')
        elif format == 'xml':
            content_type = 'application/rdf+xml'
            extension = 'rdf'
            data = st.session_state.kg_graph.serialize(format='xml')
        elif format == 'json-ld':
            content_type = 'application/ld+json'
            extension = 'jsonld'
            data = st.session_state.kg_graph.serialize(format='json-ld')
        else:
            content_type = 'text/plain'
            extension = 'txt'
            data = st.session_state.kg_graph.serialize(format='nt')
        
        # Encode the data
        b64 = base64.b64encode(data.encode()).decode()
        
        # Create download link
        href = f'<a href="data:{content_type};base64,{b64}" download="knowledge_graph.{extension}">Download as {format.upper()}</a>'
        return href
    except Exception as e:
        return f"Error creating download link: {str(e)}"

# Main app interface with tabs
tab1, tab2, tab3 = st.tabs(["Input", "Knowledge Graph", "Visualization"])

# Tab 1: Input for processing
# Tab 1: Input for processing
with tab1:
    st.header("Process Content")
    
    input_type = st.radio(
        "Select Input Type",
        ["Upload Files", "Enter URLs", "Direct Text Input"],
        horizontal=True
    )
    
    # Add annotation field that appears for file uploads and URLs
    if input_type in ["Upload Files", "Enter URLs"]:
        user_annotation = st.text_area(
            "Add text context or description (optional)",
            height=100,
            help="Provide additional context or description to help the AI understand the content"
        )
    else:
        user_annotation = None
    
    if input_type == "Upload Files":
        # File uploader for multiple files
        uploaded_files = st.file_uploader(
            "Upload files (images, audio, video, PDFs, text)",
            type=["jpg", "jpeg", "png", "gif", "mp3", "wav", "mp4", "avi", "mov", "pdf", "txt"],
            accept_multiple_files=True
        )
        
        process_button = st.button("Process Files")
        
        # Updated file upload processing with explicit media references
        if process_button and uploaded_files:
            if not st.session_state.api_initialized:
                st.error("⚠️ Please initialize the API in the sidebar first")
            else:
                progress_bar = st.progress(0)
                content_parts = []
                
                # Create a dictionary to track media files by reference ID
                media_files = {}
                
                # Track content index to reference ID mapping
                content_to_ref_map = {}
                
                # Process each file
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i / len(uploaded_files)) * 0.5)
                    
                    # Get file extension and mime type
                    file_ext = os.path.splitext(file.name)[1].lower()
                    mime_type = get_mime_type(file.name)
                    
                    # Generate a reference ID for this file
                    # Use a simple naming scheme based on file type and index
                    media_type_prefix = "txt"
                    if mime_type.startswith("image/"):
                        media_type_prefix = "img"
                    elif mime_type.startswith("audio/"):
                        media_type_prefix = "aud"
                    elif mime_type.startswith("video/"):
                        media_type_prefix = "vid"
                    elif mime_type.startswith("application/pdf"):
                        media_type_prefix = "pdf"
                        
                    # Create a reference ID that's easy for the LLM to use
                    file_ref = f"{media_type_prefix}_{normalize_string(file.name)}_{i+1}"
                    print("File ref", file_ref, file.name)
                    
                    # Store file bytes for processing
                    file_bytes = file.read()
                    
                    # Store file data in session state for persistence
                    if mime_type.startswith('image/'):
                        # For images, encode as base64 for visualization
                        encoded_image = base64.b64encode(file_bytes).decode('utf-8')
                        st.session_state.media_files_data[file_ref] = {
                            "filename": file.name,
                            "mime_type": mime_type,
                            "content": encoded_image
                        }
                    else:
                        # For non-image files
                        st.session_state.media_files_data[file_ref] = {
                            "filename": file.name,
                            "mime_type": mime_type
                        }
                    
                    # Add to media files for processing
                    media_files[file_ref] = st.session_state.media_files_data[file_ref]
                    
                    # Create content part for Gemini
                    if file_ext in ['.txt', '.pdf']:
                        # Text files
                        if file_ext == '.txt':
                            # Need to reset file pointer since we read it above
                            file.seek(0)
                            text_content = file.read().decode('utf-8')
                            content_parts.append(types.Part.from_text(text=text_content))
                        else:
                            # For PDFs, use the bytes we already read
                            content_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
                    else:
                        # Binary files (images, audio, video)
                        content_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
                        
                    # Record which content index maps to which reference ID
                    # Note: +1 because the first content part will be the prompt
                    content_to_ref_map[len(content_parts)] = file_ref
                
                progress_bar.progress(0.5)
                
                with st.spinner("Processing with Gemini..."):
                    # Process with Gemini, providing explicit media references
                    kg_data = process_with_gemini(
                        content_parts=content_parts, 
                        media_references=content_to_ref_map,
                        user_annotation=user_annotation
                    )
                    
                    if kg_data:
                        # Add to knowledge graph with media files
                        new_entities, new_relationships = add_to_knowledge_graph(kg_data, media_files)
                        
                        # Create visualization
                        html = create_visualization()
                        if html:
                            st.session_state.visualization_html = html
                        
                        # Show success message
                        st.session_state.last_processed = time.time()
                        st.success(f"✅ Successfully processed {len(uploaded_files)} files. Added {new_entities} new entities and {new_relationships} new relationships.")
                
                progress_bar.progress(1.0)
    
    elif input_type == "Enter URLs":
        # Text input for URLs
        urls_input = st.text_area(
            "Enter URLs (one per line)",
            height=100,
            help="Enter URLs for images, videos, audio files, or web pages"
        )
        
        process_button = st.button("Process URLs")
        
        # Updated URL processing with explicit media references
        if process_button and urls_input:
            if not st.session_state.api_initialized:
                st.error("⚠️ Please initialize the API in the sidebar first")
            else:
                urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
                
                if urls:
                    progress_bar = st.progress(0)
                    content_parts = []
                    
                    # Create a dictionary to track media files by reference ID
                    media_files = {}
                    
                    # Track content index to reference ID mapping
                    content_to_ref_map = {}
                    
                    # Process each URL
                    for i, url in enumerate(urls):
                        progress_bar.progress((i / len(urls)) * 0.5)
                        
                        try:
                            with st.spinner(f"Downloading from {url}..."):
                                # Download the file from URL
                                file_path = download_from_url(url)
                                
                                # Get mime type
                                mime_type = get_mime_type(file_path)
                                
                                # Generate a logical reference ID based on URL and content type
                                media_type_prefix = "web"
                                if mime_type.startswith("image/"):
                                    media_type_prefix = "img"
                                elif mime_type.startswith("audio/"):
                                    media_type_prefix = "aud"
                                elif mime_type.startswith("video/"):
                                    media_type_prefix = "vid"
                                elif mime_type.startswith("application/pdf"):
                                    media_type_prefix = "pdf"
                                elif mime_type.startswith("text/"):
                                    media_type_prefix = "txt"
                                    
                                # Create the reference ID
                                url_ref = f"{media_type_prefix}_{i+1}"
                                
                                # Store media file data in session state
                                with open(file_path, 'rb') as f:
                                    file_bytes = f.read()
                                    
                                # Prepare media data entry
                                media_data = {
                                    "url": url,
                                    "mime_type": mime_type
                                }
                                    
                                # For images, store base64 encoded data for visualization
                                if mime_type.startswith('image/'):
                                    media_data["content"] = base64.b64encode(file_bytes).decode('utf-8')
                                
                                # Store in session state
                                st.session_state.media_files_data[url_ref] = media_data
                                
                                # Add to media files for processing
                                media_files[url_ref] = media_data
                                
                                # Handle different file types for content parts
                                if mime_type.startswith('text/'):
                                    # Text files
                                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        text_content = f.read()
                                    content_parts.append(types.Part.from_text(text=text_content))
                                else:
                                    # Binary files (images, audio, video, PDFs)
                                    content_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
                                
                                # Record which content index maps to which reference ID
                                # Note: +1 because the first content part will be the prompt
                                content_to_ref_map[len(content_parts)] = url_ref
                                
                                # Clean up temporary file
                                os.unlink(file_path)
                        except Exception as e:
                            st.error(f"Error processing URL {url}: {str(e)}")
                    
                    progress_bar.progress(0.5)
                    
                    # If we have content to process
                    if content_parts:
                        with st.spinner("Processing with Gemini..."):
                            # Process with Gemini, including explicit media references
                            kg_data = process_with_gemini(
                                content_parts=content_parts, 
                                media_references=content_to_ref_map,
                                user_annotation=user_annotation
                            )
                            
                            if kg_data:
                                # Add to knowledge graph with media references
                                new_entities, new_relationships = add_to_knowledge_graph(kg_data, media_files)
                                
                                # Create visualization
                                html = create_visualization()
                                if html:
                                    st.session_state.visualization_html = html
                                
                                # Show success message
                                st.session_state.last_processed = time.time()
                                st.success(f"✅ Successfully processed {len(urls)} URLs. Added {new_entities} new entities and {new_relationships} new relationships.")
                    
                    progress_bar.progress(1.0)
    
    elif input_type == "Direct Text Input":
        # Text area for direct input
        text_input = st.text_area(
            "Enter text",
            height=200,
            help="Enter text to process"
        )
        
        process_button = st.button("Process Text")
        
        if process_button and text_input:
            if not st.session_state.api_initialized:
                st.error("⚠️ Please initialize the API in the sidebar first")
            else:
                with st.spinner("Processing with Gemini..."):
                    # Create content part
                    content_parts = [types.Part.from_text(text=text_input)]
                    
                    # Process with Gemini
                    kg_data = process_with_gemini(content_parts)
                    
                    if kg_data:
                        # Add to knowledge graph
                        new_entities, new_relationships = add_to_knowledge_graph(kg_data)
                        
                        # Create visualization
                        html = create_visualization()
                        if html:
                            st.session_state.visualization_html = html
                        
                        # Show success message
                        st.session_state.last_processed = time.time()
                        st.success(f"✅ Successfully processed text. Added {new_entities} new entities and {new_relationships} new relationships.")

# Tab 2: Knowledge Graph Details
with tab2:
    st.header("Knowledge Graph Details")
    
    # Show entity count
    st.metric("Entities", len(st.session_state.entities))
    st.metric("Relationships", len(st.session_state.relationships))
    
    # Entity and Relationship tables
    entity_tab, relationship_tab, export_tab = st.tabs(["Entities", "Relationships", "Export"])
    
    with entity_tab:
        if st.session_state.entities:
            # Create a DataFrame for display
            entity_data = []
            for entity in st.session_state.entities:
                entity_data.append({
                    "ID": entity.get("id", ""),
                    "Name": entity.get("name", ""),
                    "Type": entity.get("type", ""),
                    "Description": entity.get("description", "")
                })
            
            st.dataframe(entity_data, use_container_width=True)
        else:
            st.info("No entities in the knowledge graph yet.")
    
    with relationship_tab:
        if st.session_state.relationships:
            # Create a DataFrame for display
            relationship_data = []
            for rel in st.session_state.relationships:
                relationship_data.append({
                    "Source": rel.get("source", ""),
                    "Relationship": rel.get("type", ""),
                    "Target": rel.get("target", ""),
                    "Weight": rel.get("weight", 1.0),
                    "Description": rel.get("description", "")
                })
            
            st.dataframe(relationship_data, use_container_width=True)
        else:
            st.info("No relationships in the knowledge graph yet.")
    
    with export_tab:
        st.subheader("Export Knowledge Graph")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Download as Image")
            if st.button("Generate Image"):
                with st.spinner("Generating image..."):
                    img_buf = create_graph_image()
                    if img_buf:
                        st.markdown(get_image_download_link(img_buf, "knowledge_graph.png", "Download Knowledge Graph as PNG"), unsafe_allow_html=True)
                    else:
                        st.warning("Cannot generate image for an empty graph.")
        
        with col2:
            st.markdown("### Download as RDF")
            format_option = st.selectbox(
                "Select Format",
                ["turtle", "xml", "json-ld", "nt"],
                format_func=lambda x: {
                    "turtle": "Turtle (TTL)",
                    "xml": "RDF/XML",
                    "json-ld": "JSON-LD",
                    "nt": "N-Triples"
                }.get(x, x)
            )
            
            if st.button("Generate RDF"):
                st.markdown(get_rdf_download_link(format_option), unsafe_allow_html=True)

# Tab 3: Visualization
with tab3:
    st.header("Knowledge Graph Visualization")
    
    # Add subgraph filtering options
    if st.session_state.entities:
        with st.expander("Subgraph Filters"):
            filter_type = st.selectbox(
                "Filter Type",
                ["Full Graph", "Entity Type Filter", "Entity Neighborhood", "Minimum Weight Filter"]
            )
            
            nx_graph = graph_to_networkx()
            filter_applied = False
            
            if filter_type == "Entity Type Filter":
                # Get all entity types
                entity_types = list(set([data.get('type', 'Unknown') for _, data in nx_graph.nodes(data=True)]))
                
                selected_types = st.multiselect(
                    "Select Entity Types",
                    options=entity_types,
                    default=entity_types
                )
                
                if selected_types and selected_types != entity_types:
                    # Filter graph to only include selected types
                    nodes_to_keep = [node for node, data in nx_graph.nodes(data=True) 
                                    if data.get('type', 'Unknown') in selected_types]
                    
                    # Create subgraph
                    nx_graph = nx_graph.subgraph(nodes_to_keep)
                    filter_applied = True
            
            elif filter_type == "Entity Neighborhood":
                # Select central entity
                central_entity = st.selectbox(
                    "Select Central Entity",
                    options=[data.get('name', node) + f" ({node})" for node, data in nx_graph.nodes(data=True)]
                )
                
                # Extract entity ID from selection
                central_id = central_entity.split("(")[-1].strip(")")
                
                # Select neighborhood depth
                depth = st.slider("Neighborhood Depth", min_value=1, max_value=5, value=1)
                
                if central_id in nx_graph:
                    # Get neighbors within specified depth
                    nodes_to_keep = set([central_id])
                    current_nodes = {central_id}
                    
                    for _ in range(depth):
                        next_nodes = set()
                        for node in current_nodes:
                            next_nodes.update(nx_graph.neighbors(node))
                        nodes_to_keep.update(next_nodes)
                        current_nodes = next_nodes
                    
                    # Create subgraph
                    nx_graph = nx_graph.subgraph(nodes_to_keep)
                    filter_applied = True
            
            elif filter_type == "Minimum Weight Filter":
                # Filter relationships by minimum weight
                min_weight = st.slider(
                    "Minimum Relationship Weight",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.05
                )
                
                # Filter edges by weight
                edges_to_keep = [(u, v) for u, v, data in nx_graph.edges(data=True) 
                                if data.get('weight', 1.0) >= min_weight]
                
                # Create subgraph with filtered edges
                filtered_graph = nx.DiGraph()
                for node, data in nx_graph.nodes(data=True):
                    filtered_graph.add_node(node, **data)
                
                for u, v in edges_to_keep:
                    edge_data = nx_graph.get_edge_data(u, v)
                    filtered_graph.add_edge(u, v, **edge_data)
                
                nx_graph = filtered_graph
                filter_applied = True
            
            # Create a new visualization if filtering was applied
            if filter_applied:
                st.session_state.filtered_graph = nx_graph
                with st.spinner("Creating filtered visualization..."):
                    # Use a custom function to create visualization for the filtered graph
                    html = create_filtered_visualization(nx_graph)
                    if html:
                        st.session_state.filtered_html = html
                        st.info("Filtered graph created. Click 'Apply Filter' to view.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        refresh_button = st.button("Refresh Visualization")
        if st.session_state.entities and 'filtered_html' in st.session_state:
            apply_filter = st.button("Apply Filter")
        else:
            apply_filter = False
    
    with col2:
        # Visualization type
        if apply_filter and 'filtered_html' in st.session_state:
            current_html = st.session_state.filtered_html
        elif st.session_state.visualization_html:
            current_html = st.session_state.visualization_html
        else:
            current_html = None
    
    # Display visualization
    if current_html:
        st.components.v1.html(current_html, height=600)
        
        if refresh_button:
            with st.spinner("Refreshing visualization..."):
                html = create_visualization()
                if html:
                    st.session_state.visualization_html = html
                    st.rerun()
    else:
        if st.session_state.entities:
            # Try to create visualization
            with st.spinner("Creating visualization..."):
                html = create_visualization()
                if html:
                    st.session_state.visualization_html = html
                    st.rerun()
        else:
            st.info("No entities in the knowledge graph yet. Add content in the Input tab to create a visualization.")