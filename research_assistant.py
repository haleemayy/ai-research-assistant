
import streamlit as st
import arxiv
from transformers import pipeline

st.set_page_config(page_title="Mini Research Assistant", layout="wide")

st.title("📚 Mini AI Research Assistant")
st.subheader("Explore research topics and get paper summaries")

topic = st.text_input("Enter a research topic:", "")

if topic:
    st.info("Searching arXiv for related papers...")
    search = arxiv.Search(
        query=topic,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = list(search.results())

    if not papers:
        st.warning("No papers found. Try a different topic.")
    else:
        paper = papers[0]
        st.success(f"Found paper: {paper.title}")
        st.markdown(f"**Authors**: {', '.join([a.name for a in paper.authors])}")
        st.markdown(f"**Published**: {paper.published.strftime('%Y-%m-%d')}")
        st.markdown(f"**Link**: [View on arXiv]({paper.entry_id})")
        st.markdown(f"**Abstract**: {paper.summary[:1000]}...")

        st.info("Generating summary using HuggingFace model...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        summary = summarizer(paper.summary, max_length=150, min_length=60, do_sample=False)[0]['summary_text']

        st.markdown("### ✨ Summary:")
        st.write(summary)

        st.markdown("### 💡 Suggested Research Ideas:")
        idea_prompt = f"What could be future research directions based on this paper about {topic}?"
        research_ideas = summarizer(idea_prompt, max_length=80, min_length=40, do_sample=True)[0]['summary_text']
        st.write(research_ideas)
