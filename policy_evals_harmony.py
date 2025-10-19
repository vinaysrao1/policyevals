import os
import json
import re
import argparse
from datetime import datetime
from huggingface_hub import login
from vllm import LLM, SamplingParams
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Message,
    Role,
    Conversation,
    SystemContent,
    DeveloperContent
)

# Hugging Face authentication
token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN is not set")
login(token=token, add_to_git_credential=False)

# Ensure token is available for transformers/vllm
os.environ["HUGGING_FACE_HUB_TOKEN"] = token
os.environ["HF_TOKEN"] = token


def read_file(filepath: str) -> str:
    """Read content from a file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_model(model_id: str = "openai/gpt-oss-20b") -> LLM:
    """Load the vLLM model"""
    print(f"Loading model: {model_id}")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        dtype="auto",
        tensor_parallel_size=1
    )
    return llm


def generate_response(llm: LLM, encoding, system_content: str, developer_content: str, user_content: str) -> tuple[str, str]:
    """Generate response using vLLM with Harmony encoding

    Returns:
        tuple: (full_response, final_channel_content)
    """
    # Build conversation with Harmony format using proper structured content
    conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(developer_content)
        ),
        Message.from_role_and_content(Role.USER, user_content)
    ])

    # Render conversation to token IDs
    prefill_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    # Get stop tokens for assistant
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    # Configure sampling
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=500,
        stop_token_ids=stop_token_ids
    )

    # Generate response
    # vLLM expects prompts parameter with token_ids key
    from vllm.inputs import TokensPrompt
    prompt = TokensPrompt(prompt_token_ids=prefill_ids)
    outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
    completion_token_ids = outputs[0].outputs[0].token_ids

    # Get the raw text output as well for debugging
    raw_text = outputs[0].outputs[0].text

    # Try to parse response back to messages
    try:
        response_messages = encoding.parse_messages_from_completion_tokens(
            completion_token_ids,
            Role.ASSISTANT
        )

        # Extract the full response as text
        full_response = ""
        for msg in response_messages:
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            full_response += f"<|channel|>{msg.channel}<|message|>{content_str}"

        # Extract content from the final channel
        final_content = ""
        for msg in response_messages:
            if msg.channel == "final":
                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                final_content = content_str.strip()
                break

        if not final_content and response_messages:
            content_str = response_messages[0].content if isinstance(response_messages[0].content, str) else str(response_messages[0].content)
            final_content = content_str.strip()
    except Exception as e:
        # If parsing fails, fall back to raw text
        print(f"Warning: Failed to parse Harmony format: {e}")
        print(f"Falling back to raw text output")
        # Ensure raw_text is a string
        full_response = raw_text if isinstance(raw_text, str) else str(raw_text)
        final_content = full_response

    return full_response, final_content


def extract_label(response: str) -> str:
    """Extract label from model response (expects '0' or '1')"""
    # Look for 0 or 1 in the response
    label_match = re.search(r'\b([01])\b', response)
    return label_match.group(1) if label_match else None


def build_system_message() -> str:
    """Build the system message for Harmony format"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_message = f"""You are a content moderation assistant.
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: low

# Valid channels: analysis, commentary, final."""
    return system_message


def build_developer_message(policy_def: str) -> str:
    """Build the developer message with instructions and policy"""
    developer_message = """# Instructions

You are a content moderator. You will read a given policy and classify user prompts based on that policy.

You must respond very concisely with an answer in the following format: <answer>response</answer>

Rules:
- Only respond with 1 or 0
- Respond with 1 if the policy applies to the user prompt
- Respond with 0 if the policy does not apply to the user prompt
- Provide your response in the final channel

Read the following policy carefully:

""" + policy_def
    return developer_message


def build_user_message(content: str) -> str:
    """Build the user message with the content to classify"""
    user_message = f"""Classify the following user prompt: {content}

Remember to respond in the final channel.
Use 1 if policy applies, 0 if policy does not apply."""
    return user_message


def main():
    """Main function to run policy evaluation"""
    parser = argparse.ArgumentParser(description="Run policy-based content evaluation with vLLM")
    parser.add_argument("--policy", required=True, help="Path to policy file")
    parser.add_argument("--evals", required=True, help="Path to evaluation set file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model ID (default: openai/gpt-oss-20b)")
    args = parser.parse_args()

    # Load model
    llm = load_model(args.model)

    # Load Harmony encoding
    print("Loading Harmony encoding...")
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Read policy and evaluation data
    policy_def = read_file(args.policy)
    evals_set = read_file(args.evals)

    # Build Harmony format messages
    system_message = build_system_message()
    developer_message = build_developer_message(policy_def)

    print(f"Starting evaluation... Writing results to: {args.output}")

    # Process each line in the evaluation set
    with open(args.output, 'w', encoding='utf-8') as out_f:
        for i, line in enumerate(evals_set.splitlines()):
            if line and line.strip():
                content = line.strip()
                user_message = build_user_message(content)
                full_response, final_response = generate_response(
                    llm, encoding, system_message, developer_message, user_message
                )
                label = extract_label(final_response)

                result = {
                    "index": i,
                    "content": content,
                    "label": label,
                    "final_response": final_response,
                    "full_response": full_response,
                    "timestamp": datetime.now().isoformat()
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f"{i}: Label={label}")

    print(f"Evaluation complete. Results written to: {args.output}")


if __name__ == "__main__":
    main()
