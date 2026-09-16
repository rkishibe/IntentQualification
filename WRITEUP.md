# 3.1 Approach

**Describe your system architecture.**

What components does it include? \
Architecture has been fragmented into responsiblities and stages in the classificaition process as follows: data classes, data visualization, embedding client, LLM client and Orchestrator. Additionally an utils.py file contains recurrent utility functions used by these classes.

How do they interact? \
The responsibilites are isolated\

Why did you choose this design?\



# 3.2 Tradeoffs

What did you optimize for?

Speed: prefferably under 10 minutes.
Cost: model ran locally

Examples:

    speed
    cost
    accuracy
    simplicity
    robustness

What trade-offs did you intentionally make? \
I prioritised speed and cost in favour of fine-granulation of the query prompts. This choice has been taken on account of the limited sample data size.
# 3.3 Error Analysis

Where does your system struggle? \


Show concrete examples of companies it misclassifies and explain why.
# 3.4 Scaling

If the system needed to handle 100,000 companies per query instead of 500, what would you change?
# 3.5 Failure Modes

When might your system produce confident but incorrect results? \
In development I encountered incorrect parsing of NAICS codes that produced false positives. Due to limited capabilities, the LLM prompt outputs only the first two digits of the NAICS code which point to a larger field. A possible consequence would be that highly detailed domains would be aprsed incorrectly/sent as semantic queries instead of structured, making the query time take longer.

What would you monitor in production to detect these failures?