"""Example: how should a Kigali loyalty app sign its first 100 merchants?"""

import asyncio
import os

from cdm import (
    Constraint,
    Dimension,
    JevJudge,
    MockJudge,
    Option,
    Problem,
    decide,
    judge_problem,
)

problem = Problem(
    goal="Sign the first 100 paying merchants for a mobile loyalty/membership app within 90 days",
    context={
        "city": "Kigali, Rwanda",
        "team": "3 founders, small budget (about $15k for go-to-market)",
        "merchants": "cafes, salons, small restaurants, boutiques; most take MTN MoMo payments",
    },
    precedents=[
        "Door-to-door sales visits with a printed brochure",
        "Paid social media ads targeting business owners",
        "Free trial for the first three months",
    ],
    options=[
        Option(
            "door",
            "Founders walk into shops and pitch the app with a printed brochure, offering a free 3-month trial.",
            baseline=True,
        ),
        Option("ads", "Run Instagram and Facebook ads aimed at small business owners in Kigali."),
        Option(
            "momo",
            "Partner with MoMo agents: pay each agent a commission for every merchant they sign up during their normal visits.",
        ),
        Option(
            "street",
            "Pick one busy street (e.g. KN 4 Ave), sign every merchant on it, and launch a street-wide shared reward card customers can use at any shop on the street.",
        ),
        Option(
            "reverse",
            "Recruit 2,000 customers first with a free 'Kigali passport' of discounts, then show merchants the waiting list of their own customers.",
        ),
        Option(
            "fake",
            "Show merchants a list of well-known shops claiming they already joined, before they actually have, to create urgency.",
        ),
    ],
    dimensions=[
        Dimension(
            "speed",
            "How quickly would `option` plausibly sign paying merchants?",
            (
                "Almost no merchants in 90 days",
                "A handful",
                "About a quarter of the target",
                "About half the target",
                "The full target or more",
            ),
            weight=0.4,
        ),
        Dimension(
            "cost",
            "How affordable is `option` for the team's budget?",
            (
                "Far beyond the budget",
                "Uses all the budget",
                "Uses about half",
                "Uses a small part",
                "Almost free",
            ),
            weight=0.2,
        ),
        Dimension(
            "moat",
            "How hard would it be for a competitor to copy the advantage `option` creates?",
            (
                "Trivial to copy",
                "Easy to copy",
                "Takes some effort",
                "Hard: builds network effects or relationships",
                "Very hard: a lasting structural advantage",
            ),
            weight=0.25,
        ),
        Dimension(
            "retention",
            "How likely are merchants won by `option` to keep paying after six months?",
            ("Very unlikely", "Unlikely", "Even odds", "Likely", "Very likely"),
            weight=0.15,
        ),
    ],
    constraints=[
        Constraint("deceptive", "deceive merchants or customers", hard=True),
        Constraint(
            "legal", "break Rwandan consumer-protection or financial regulations", hard=True
        ),
        Constraint(
            "founders", "require more than the three founders' full-time effort", hard=False
        ),
    ],
    niches={
        "direct": "Founders sell directly to individual merchants",
        "channel": "A third party with existing merchant relationships sells for us",
        "network": "Creates a cluster or network where each merchant makes the app more valuable to others",
        "demand": "Builds customer demand first so merchants come to us",
        "paid": "Buys attention through advertising",
    },
)


async def main() -> None:
    judges = [JevJudge()] if os.environ.get("TYPESAFE_API_KEY") else [MockJudge()]
    print("judges:", [j.name for j in judges])
    judgments = await judge_problem(problem, judges)
    d = decide(problem, judgments)
    print(
        f"\n{'option':8} {'E[U]':>5} {'sd':>5} {'CVaR':>5} {'nov':>5} {'Pbest':>5} {'EI':>6}  rel  {'status':7} niche / checks"
    )
    for s in sorted(d.stats, key=lambda s: -s.mean):
        print(
            f"{s.id:8} {s.mean:5.2f} {s.std:5.2f} {s.cvar:5.2f} {s.novelty:5.2f} {s.p_best:5.2f} "
            f"{s.ei:6.3f}  {'yes' if s.reliable else ' no'}  {s.status:7} {s.niche} {s.checks or ''}"
        )
    shares = sorted(d.portfolio.items(), key=lambda kv: -kv[1])
    print(f"\ncommit:     {d.commit}")
    print(f"innovation: {d.innovation}")
    print(f"portfolio:  {', '.join(f'{k} {v:.0%}' for k, v in shares)}")
    esc = f"  -> escalate {d.escalate} to a reasoning judge" if d.escalate else ""
    print(f"EVPI:       {d.evpi:.3f}{esc}")


if __name__ == "__main__":
    asyncio.run(main())
