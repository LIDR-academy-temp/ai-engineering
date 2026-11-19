ESTIMATION_EXAMPLES: list[dict] = [
    {
        "meeting_summary": (
            "The client needs an inventory management web platform for a mid-size warehouse "
            "operation. They want real-time stock tracking, automated reorder alerts when items "
            "fall below a configurable threshold, a role-based access system (admin, warehouse "
            "manager, viewer), CSV/Excel import and export of inventory data, and a dashboard "
            "with key metrics (turnover rate, stock value, low-stock items). They already have "
            "a PostgreSQL database running in AWS that we should integrate with. No mobile app "
            "is needed — the warehouse team will use tablets with the web interface."
        ),
        "estimation": """\
## Inventory Management Web Platform

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Requirements analysis and technical design | 16 | 1,000 |
| Database schema design and migrations | 12 | 750 |
| Authentication and role-based access control | 20 | 1,250 |
| Product and stock CRUD API | 24 | 1,500 |
| Automated reorder alert engine | 16 | 1,000 |
| CSV/Excel import and export module | 14 | 875 |
| Dashboard with key metrics and charts | 20 | 1,250 |
| Frontend: inventory views and search/filter | 24 | 1,500 |
| Frontend: admin panel and user management | 12 | 750 |
| AWS PostgreSQL integration and deployment | 10 | 625 |
| Testing (unit, integration, E2E) | 20 | 1,250 |
| Code review, QA, and bug fixing | 12 | 750 |

### Totals

- **Total hours:** 200
- **Total cost:** 12,500 EUR

### Recommended Team

- 1 Senior Backend Developer (lead)
- 1 Mid-level Full-Stack Developer
- 1 QA Engineer (part-time, last 3 weeks)

### Estimated Duration

**10 weeks** with a two-person development team.""",
    },
    {
        "meeting_summary": (
            "A real estate agency wants a high-conversion landing page to capture leads for "
            "luxury property listings. The page should feature a hero section with a video "
            "background, a curated property gallery with filtering by price range and location, "
            "a lead capture form, and client testimonials. All leads must be pushed in real-time "
            "to their existing HubSpot CRM via the API. The design must be responsive, "
            "mobile-first, and follow their brand guidelines (they will provide a Figma file). "
            "They also want basic analytics integration with Google Tag Manager and Meta Pixel "
            "for ad campaign tracking."
        ),
        "estimation": """\
## Real Estate Landing Page with CRM Integration

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Requirements review and UX analysis | 8 | 500 |
| UI design adaptation from Figma to code | 16 | 800 |
| Hero section with video background | 8 | 500 |
| Property gallery with filtering | 16 | 1,000 |
| Lead capture form with validation | 8 | 500 |
| HubSpot CRM API integration | 14 | 875 |
| Testimonials carousel component | 6 | 375 |
| Google Tag Manager and Meta Pixel setup | 6 | 375 |
| Responsive design and cross-browser testing | 10 | 625 |
| Performance optimization (Core Web Vitals) | 8 | 500 |
| Deployment and DNS configuration | 4 | 250 |

### Totals

- **Total hours:** 104
- **Total cost:** 6,300 EUR

### Recommended Team

- 1 Senior Frontend Developer
- 1 UI/UX Designer (part-time, first 2 weeks)

### Estimated Duration

**5 weeks** with a single frontend developer and part-time design support.""",
    },
    {
        "meeting_summary": (
            "A B2B startup is building a SaaS platform to manage software subscriptions for "
            "SMEs. Core features include: user registration and company onboarding, a dashboard "
            "showing all active subscriptions with renewal dates and monthly spend, the ability "
            "to add/edit/cancel subscriptions, integration with Stripe for payment processing "
            "and invoice generation, a notification system (email alerts 30/7/1 days before "
            "renewal), and an admin panel to manage customer accounts. They want a REST API "
            "with a React frontend. MVP scope — they plan to iterate after launch."
        ),
        "estimation": """\
## SaaS Subscription Management Platform (MVP)

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Architecture design and project setup | 12 | 750 |
| User registration and authentication (JWT) | 16 | 1,000 |
| Company onboarding flow | 12 | 750 |
| Subscription CRUD API and data model | 20 | 1,250 |
| Dashboard: active subscriptions and spend analytics | 24 | 1,500 |
| Stripe integration: payments and invoices | 28 | 1,750 |
| Email notification system (renewal reminders) | 16 | 1,000 |
| Admin panel for customer management | 20 | 1,250 |
| React frontend: views, forms, and routing | 40 | 2,500 |
| API documentation (OpenAPI/Swagger) | 6 | 375 |
| Testing (unit, integration, Stripe sandbox) | 24 | 1,500 |
| Deployment, CI/CD pipeline, and staging env | 16 | 1,000 |
| Security audit and hardening | 10 | 625 |

### Totals

- **Total hours:** 244
- **Total cost:** 15,250 EUR

### Recommended Team

- 1 Senior Full-Stack Developer (lead)
- 1 Mid-level Backend Developer
- 1 Mid-level Frontend Developer
- 1 QA Engineer (part-time, last 4 weeks)

### Estimated Duration

**12 weeks** with a three-person core development team.""",
    },
    {
        "meeting_summary": (
            "The client needs a web portal where customers can sign in, view their invoices, "
            "and download them as PDFs. An existing billing API provides invoice data and PDFs. "
            "The portal must only show invoices that belong to the signed-in customer."
        ),
        "estimation": """\
## Customer Invoice Portal

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Requirements analysis and technical design | 12 | 750 |
| UI/UX design for sign-in and invoice views | 8 | 400 |
| Customer sign-in and invoice access controls | 16 | 1,000 |
| Integration with the existing billing API | 18 | 1,125 |
| Invoice list and PDF download | 20 | 1,250 |
| Integration and access-control tests | 12 | 750 |
| QA and bug fixing | 10 | 625 |
| Deployment and handover documentation | 8 | 500 |

### Totals

- **Total hours:** 104
- **Total cost:** 6,400 EUR

### Recommended Team

- 1 Full-Stack Developer
- 1 UI/UX Designer (part-time during the first week)

### Estimated Duration

**3–4 weeks**, assuming billing API credentials and a test environment are available
at project kickoff.""",
    },
]


def format_examples_for_prompt(examples: list[dict]) -> str:
    """Format estimation examples into a string suitable for injection into a system prompt."""
    parts: list[str] = []
    for i, example in enumerate(examples, start=1):
        parts.append(
            f"--- EXAMPLE {i} ---\n"
            f"Meeting Summary:\n{example['meeting_summary']}\n\n"
            f"Estimation:\n{example['estimation']}\n"
        )
    return "\n".join(parts)
