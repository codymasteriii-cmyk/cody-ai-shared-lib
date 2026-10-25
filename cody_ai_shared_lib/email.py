"""Email delivery — thin Resend wrapper, shared across Cody AI projects.

Extracted from NewsIntelligence's backend/app/api/delivery.py and
ingestion.py's _send_completion_email(), which both independently read
RESEND_API_KEY, built a sender/recipient pair from env vars with the same
fallback shape, called resend.Emails.send(...), and logged failures the
same way. That's the entire generic surface — everything else in those two
functions (which articles to include, the health-check queries, the HTML
table of Jev disagreements, a job board digest's per-posting rows) is
domain content, not an email-sending concern, and stays in each app.

Deliberately NOT included here, same reasoning as auth.py's own boundary:
- Which env var names hold the sender/recipient (DELIVERY_SENDER_EMAIL /
  DELIVERY_USER_EMAIL is NewsIntelligence's own naming — JobSearch's
  ADMIN_NOTIFY_EMAIL is a different name for a similar idea). Callers pass
  `to` and `sender` as plain strings; reading whichever env var names an
  app happens to use is that app's own few lines, not a shared concern.
- Any HTML templating. send_email() takes a finished html string.

Usage:
    from cody_ai_shared_lib.email import send_email, EmailError

    try:
        message_id = send_email(
            to=os.getenv("ADMIN_NOTIFY_EMAIL"),
            subject="New JobSearch sign-in: needs a role",
            html="<p>...</p>",
            sender="JobSearch Copilot <onboarding@resend.dev>",
        )
    except EmailError as e:
        logger.error(f"Notification email failed: {e}")
"""
import logging
import os

import resend
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

logger = logging.getLogger("cody-ai-shared-email")


class EmailError(Exception):
    """Raised when RESEND_API_KEY is missing or Resend's API call fails."""
    pass


def send_email(to: str, subject: str, html: str, sender: str) -> str:
    """Send one HTML email via Resend. Returns Resend's message id.

    Always raises EmailError on failure (missing API key, or whatever
    Resend's client raises) rather than offering a swallow-and-log mode —
    callers that want to swallow a failed notification without affecting
    their own request/job (NewsIntelligence's ingestion.py does, since it's
    a background job where a notification failure shouldn't fail the run;
    its delivery.py doesn't, since there the email *is* the response)
    already wrap their own call in try/except, and should keep doing that
    here rather than this function guessing which behavior a given caller
    wants.
    """
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        raise EmailError("RESEND_API_KEY is not set")
    resend.api_key = api_key

    try:
        result = resend.Emails.send({
            "from": sender,
            "to": to,
            "subject": subject,
            "html": html,
        })
    except Exception as e:
        logger.error(f"[Email] Send failed ({subject!r} to {to}): {type(e).__name__} - {e}")
        raise EmailError(f"Resend send failed: {e}") from e

    message_id = result.get("id")
    logger.info(f"[Email] Sent {subject!r} to {to} (id={message_id})")
    return message_id
