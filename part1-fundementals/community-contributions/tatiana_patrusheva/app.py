import os
from datetime import datetime, timedelta, timezone

import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. Add it to your .env file or deployment secrets."
    )

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------------------------------------------------
# Reddit post retrieval
# ------------------------------------------------------------

def subreddit_search(
    subreddit: str,
    date: str = "yesterday",
    limit: int = 100
) -> str:
    """
    Fetches Reddit posts from a specific subreddit and returns posts from the requested date,
    including title, text, score, number of comments, creation time, and URL.

    Args:
        subreddit (str): Subreddit name without r/, for example "MicrosoftFabric".
        date (str): Date to search for. Supports "yesterday" or YYYY-MM-DD.
        limit (int): Number of latest posts to check. Uses pagination if limit > 100.

    Returns:
        str: A formatted string containing matching Reddit posts.
    """

    subreddit = subreddit.replace("r/", "").strip()

    if not subreddit:
        return "No subreddit was provided."

    # Reddit returns timestamps in UTC, so we compare dates in UTC.
    now = datetime.now(timezone.utc)

    if date.lower().strip() == "yesterday":
        target_start = datetime(
            year=now.year,
            month=now.month,
            day=now.day,
            tzinfo=timezone.utc
        ) - timedelta(days=1)
    else:
        target_start = datetime.strptime(date.strip(), "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    target_end = target_start + timedelta(days=1)

    headers = {
        # User-Agent helps Reddit identify your script.
        # It is not authentication, but Reddit may reject requests without it.
        "User-Agent": "agentic-ai-lab-subreddit-summarizer/0.1 by Tatiana"
    }

    matching_posts = []
    fetched_count = 0
    after = None

    # Reddit usually returns max 100 posts per request.
    # This loop allows the app to check more than 100 posts if the user increases the limit.
    while fetched_count < limit:
        batch_size = min(100, limit - fetched_count)

        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={batch_size}"

        if after:
            url += f"&after={after}"

        response = requests.get(url, headers=headers, timeout=20)

        # Give a clearer message for common Reddit/API issues.
        if response.status_code == 404:
            return f"Subreddit r/{subreddit} was not found."
        if response.status_code == 403:
            return f"Access to r/{subreddit} is forbidden. The subreddit may be private or restricted."
        if response.status_code == 429:
            return "Reddit rate limit reached. Please wait a bit and try again."

        response.raise_for_status()

        data = response.json()
        children = data.get("data", {}).get("children", [])

        if not children:
            break

        for item in children:
            post = item.get("data", {})

            post_time = datetime.fromtimestamp(
                post.get("created_utc", 0),
                tz=timezone.utc
            )

            if target_start <= post_time < target_end:
                matching_posts.append({
                    "title": post.get("title", ""),
                    "text": post.get("selftext", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "url": "https://www.reddit.com" + post.get("permalink", ""),
                    "created_utc": post_time.strftime("%Y-%m-%d %H:%M UTC")
                })

        fetched_count += len(children)
        after = data.get("data", {}).get("after")

        if not after:
            break

    if not matching_posts:
        return f"No posts found in r/{subreddit} for {date}."

    # Sort by discussion level first, then by score.
    matching_posts = sorted(
        matching_posts,
        key=lambda post: (post["num_comments"], post["score"]),
        reverse=True
    )

    formatted_posts = []

    for i, post in enumerate(matching_posts, start=1):
        # Truncate very long post text to avoid sending too much text to OpenAI.
        text = post["text"]

        if len(text) > 1000:
            text = text[:1000] + "... [truncated]"

        formatted_posts.append(
            f"""
Post {i}
Title: {post['title']}
Text: {text}
Score: {post['score']}
Comments: {post['num_comments']}
Created: {post['created_utc']}
URL: {post['url']}
"""
        )

    return "\n\n".join(formatted_posts)


# ------------------------------------------------------------
# OpenAI summarization
# ------------------------------------------------------------

def summarize_subreddit_posts(
    subreddit: str,
    date: str = "yesterday",
    preferences: str = "",
    limit: int = 100
) -> str:
    """
    Retrieves Reddit posts from a subreddit and summarizes them using OpenAI.
    """

    posts = subreddit_search(
        subreddit=subreddit,
        date=date,
        limit=limit
    )

    if (
        posts.startswith("No posts found")
        or posts.startswith("Subreddit")
        or posts.startswith("Access")
        or posts.startswith("Reddit rate limit")
        or posts.startswith("No subreddit")
    ):
        return posts

    # Avoid sending extremely large text to the model.
    max_chars = 30000

    if len(posts) > max_chars:
        posts = posts[:max_chars] + "\n\n[Additional posts were truncated to keep the summary within token limits.]"

    system_message = """
You are a Reddit subreddit summarization assistant.

Your job is to summarize Reddit posts retrieved from a subreddit.

Rules:
- Summarize only the information provided in the retrieved posts.
- Do not invent posts, comments, links, opinions, or trends.
- If the available posts are limited, mention that clearly.
- Keep the summary concise, structured, and easy to read.
- Use the user's preferences only to decide what to emphasize in the summary.
- Give more attention to posts with higher comment counts and higher scores.
"""

    user_message = f"""
Subreddit: r/{subreddit}
Date: {date}
User preferences for the summary: {preferences}

Retrieved posts:
{posts}

Please provide:
1. A short overall summary
2. Main topics discussed
3. Most discussed posts
4. Posts with notable scores
5. Repeated issues, questions, or themes
6. A brief takeaway

When relevant, emphasize the user's preferences:
{preferences}

Be concise and clear.
Only return the final subreddit summary.
Do not explain the tool call process.
"""

    openai_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    return openai_response.choices[0].message.content


# ------------------------------------------------------------
# Gradio styling
# ------------------------------------------------------------

custom_css = """
:root {
    --reddit-orange: #ff4500;
    --reddit-orange-dark: #d93a00;
    --reddit-bg: #fff7f3;
    --reddit-card: #ffffff;
    --reddit-text: #1c1c1c;
    --reddit-muted: #6b7280;
}

.gradio-container {
    background: linear-gradient(135deg, #fff7f3 0%, #fff1eb 45%, #ffffff 100%) !important;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app-container {
    max-width: 980px;
    margin: 0 auto;
}

#hero {
    background: linear-gradient(135deg, #ff4500 0%, #ff7a1a 100%);
    color: white;
    padding: 32px;
    border-radius: 24px;
    box-shadow: 0 18px 40px rgba(255, 69, 0, 0.22);
    margin-bottom: 24px;
}

#hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

#hero p {
    font-size: 17px;
    opacity: 0.95;
    margin-bottom: 0;
}

.input-card, .output-card {
    background: white;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 30px rgba(17, 24, 39, 0.08);
    border: 1px solid rgba(255, 69, 0, 0.12);
}

#submit-btn {
    background: linear-gradient(135deg, #ff4500 0%, #ff7a1a 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 12px 18px !important;
    box-shadow: 0 10px 22px rgba(255, 69, 0, 0.28) !important;
}

#submit-btn:hover {
    background: linear-gradient(135deg, #d93a00 0%, #ff6500 100%) !important;
    transform: translateY(-1px);
}

#clear-btn {
    border-radius: 16px !important;
}

.gr-textbox textarea,
.gr-textbox input {
    border-radius: 14px !important;
}

.gr-slider {
    border-radius: 14px !important;
}

#tips {
    background: #fff1eb;
    border-left: 5px solid #ff4500;
    padding: 14px 18px;
    border-radius: 16px;
    color: #3a1d12;
    margin-top: 12px;
}

footer {
    visibility: hidden;
}
"""


# ------------------------------------------------------------
# Gradio wrapper
# ------------------------------------------------------------

def gradio_summarize_subreddit(subreddit, date, preferences, limit):
    """
    Wrapper function used by the Gradio interface.
    Handles user input validation and returns the final summary.
    """

    try:
        if not subreddit or not subreddit.strip():
            return "Please enter a subreddit name, for example `MicrosoftFabric`."

        if not date or not date.strip():
            date = "yesterday"

        return summarize_subreddit_posts(
            subreddit=subreddit,
            date=date,
            preferences=preferences,
            limit=int(limit)
        )

    except ValueError as e:
        return f"""
### Date format issue

Please use either:

- `yesterday`
- `YYYY-MM-DD`, for example `2026-05-18`

Error details: `{str(e)}`
"""

    except Exception as e:
        return f"""
### Something went wrong

Error details:

`{str(e)}`
"""


# ------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------

with gr.Blocks(
    css=custom_css,
    title="Reddit Subreddit Summarizer",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="slate"
    )
) as demo:

    with gr.Column(elem_id="app-container"):

        gr.HTML(
            """
            <div id="hero">
                <h1>🔥 Reddit Subreddit Summarizer</h1>
                <p>
                    Pick a subreddit, choose a date, and get a concise AI-powered summary of the most discussed posts.
                </p>
            </div>
            """
        )

        with gr.Row():

            with gr.Column(scale=1, elem_classes="input-card"):
                gr.Markdown("## Search settings")

                subreddit_input = gr.Textbox(
                    label="Subreddit",
                    value="MicrosoftFabric",
                    placeholder="Example: MicrosoftFabric, PowerBI, datascience",
                    info="Enter the subreddit name without r/"
                )

                date_input = gr.Textbox(
                    label="Date",
                    value="yesterday",
                    placeholder="yesterday or 2026-05-18",
                    info="Use 'yesterday' or a date in YYYY-MM-DD format"
                )

                preferences_input = gr.Textbox(
                    label="Summary preferences",
                    value="Focus on Power BI, Fabric, semantic models, data engineering, common issues, and highly discussed posts.",
                    placeholder="Example: focus on technical issues, questions, complaints, tutorials, or highly discussed posts",
                    lines=5,
                    info="These preferences are used only by OpenAI when creating the summary"
                )

                limit_input = gr.Slider(
                    label="Number of latest posts to check",
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    info="Higher values check more posts but may take longer"
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "Summarize Subreddit 🚀",
                        elem_id="submit-btn",
                        scale=2
                    )

                    clear_btn = gr.ClearButton(
                        components=[
                            subreddit_input,
                            date_input,
                            preferences_input
                        ],
                        value="Clear",
                        elem_id="clear-btn",
                        scale=1
                    )

                gr.HTML(
                    """
                    <div id="tips">
                        <strong>Tip:</strong> If the subreddit is very active, increase the post limit to 300–500.
                        If it is quiet, 100 is usually enough.
                    </div>
                    """
                )

            with gr.Column(scale=2, elem_classes="output-card"):
                gr.Markdown("## Summary")

                output = gr.Markdown(
                    value="Your subreddit summary will appear here.",
                    label="Subreddit Summary"
                )

        gr.Examples(
            examples=[
                [
                    "MicrosoftFabric",
                    "yesterday",
                    "Focus on Power BI, Fabric, semantic models, data engineering, common issues, and highly discussed posts.",
                    100
                ],
                [
                    "PowerBI",
                    "yesterday",
                    "Focus on user problems, dashboard performance, DAX, semantic models, and practical tips.",
                    200
                ],
                [
                    "datascience",
                    "yesterday",
                    "Focus on career advice, project ideas, machine learning, and beginner questions.",
                    200
                ]
            ],
            inputs=[
                subreddit_input,
                date_input,
                preferences_input,
                limit_input
            ],
            label="Try an example"
        )

        submit_btn.click(
            fn=gradio_summarize_subreddit,
            inputs=[
                subreddit_input,
                date_input,
                preferences_input,
                limit_input
            ],
            outputs=output,
            show_progress="full"
        )


# Queue enables better request handling and visible loading/progress behaviour.
demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )