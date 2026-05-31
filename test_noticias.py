import asyncio
import json
import os
from server import fetch_trends_and_news, write_content, stage_for_approval

async def main():
    print("1. Buscando notícias sobre Construção Civil...")
    news_json = fetch_trends_and_news("construção civil mercado imobiliário", max_results=3)
    news_data = json.loads(news_json)
    
    if "error" in news_data:
        print("Erro na busca:", news_data["error"])
        return
        
    print(f"Foram encontradas {len(news_data.get('news', []))} notícias.")
    print("2. Passando as notícias para o Redator-Chefe (Douglas Adams style)...")
    
    # Extract text to pass to LLM
    raw_text = ""
    for n in news_data.get('news', []):
        raw_text += f"- Título: {n['title']}\n  Trecho: {n['snippet']}\n  Fonte: {n['source']} ({n['date']})\n\n"
        
    print(raw_text)
    
    content_json = await write_content("noticia", raw_text)
    content_data = json.loads(content_json)
    
    html_content = content_data.get("html_generated", "")
    
    if not html_content:
        print("Erro ao gerar o HTML.")
        return
        
    print("3. Artigo gerado com sucesso! Salvando nos rascunhos...")
    
    # We will generate a title dynamically from the news, or just use a default one for the test
    title = "Mercado em Alta: A Realidade Crua do Canteiro"
    
    result_json = stage_for_approval(title, html_content, "noticia")
    result_data = json.loads(result_json)
    
    print("Concluído!")
    print(result_data["message"])

if __name__ == "__main__":
    asyncio.run(main())
