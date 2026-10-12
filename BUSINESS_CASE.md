# Business case

## The problem on the line

A fault on an assembly line rarely comes alone. A safety gate opens, the robot behind it stops, the conveyor backs up, and every station writes its own alarm. The alarm log shows these as separate rows, thousands of them per day.

During a line stop, the maintenance engineer reads that log to find out what happened first. Every extra row is something to rule out while the line is standing still. The cost is downtime: the longer it takes to find the first fault, the longer the stop.

## Who uses it

| Who | What they need |
|---|---|
| Maintenance and shift engineers | Which alarms belong together, and which one came first, during and after a stop |
| Production and process engineers | Fault patterns that keep coming back on a station |
| Plant IT | Something that runs on the plant's servers every day without supervision |

## What the application gives them

- Every alert tagged with the group of alerts that fired with it (Model 1), the kind of message it is (Model 2) and a coarse category for triage (Model 3)
- One clustered file per PLC and day, ready for a dashboard or a report
- A multi-PLC view that shows groups spanning several controllers
- A run log (`Logs.csv`) that shows what was processed and what failed
- Offline operation on plant servers, with models that are updated each day instead of retrained

## Evidence so far

| Signal | Result |
|---|---|
| Review by Volkswagen's engineers, single PLC | Groups were readable and matched failure modes they knew |
| Review by Volkswagen's engineers, multi-PLC | Some groups linked PLCs that sit far apart in the plant, so cross-machine links need the plant layout |
| Deployment | Packaged and installed for offline use on Windows servers |
| Generated data with hidden labels | Scores per model in the [README](README.md) and the notebooks |

## How I would measure it in a pilot

A line can't be split into random test and control groups, so I would run it on one line section and compare shifts that use the grouped view with shifts that don't, over the same weeks.

**Primary KPI**
- **Time to root cause** for line stops: from the first alarm to the cause recorded in the shift log

**Secondary KPIs**
- **Repair time** (MTTR) for stops that involve several alarms
- **Rows to read per incident**: alarms in the raw log versus groups shown to the engineer
- **Recurring groups with a maintenance action**: how many repeated patterns lead to a fix

**Guardrails**
- Groups that engineers mark as wrong
- Root causes the grouping missed
- The daily run finishes before the morning shift, with no failed files in `Logs.csv`

**Before a pilot**
- Check the Model 1 settings per PLC with the engineers on a sample of past days
- Use multi-PLC mode only once the plant layout is part of it

## Rollout path

1. Single-PLC mode on the busiest PLCs, one run per day.
2. A short morning review of the largest groups on the dashboard.
3. Plant layout added to multi-PLC mode, so only physically connected PLCs can be linked.
4. Shorter cycles, from daily files to hourly files, once the daily run is stable.

## Risks

| Risk | How it is handled |
|---|---|
| Unrelated alerts grouped because a station is busy, or one device repeats a message | Settings tuned per PLC with the engineers; the dashboard shows each group's messages |
| Links between PLCs that can't affect each other | Plant layout as a constraint in multi-PLC mode |
| Alarm texts full of abbreviations nobody has documented | Model 2 groups messages by their text alone, without a dictionary |
| New stations or new messages over time | Models are updated with every run; retrained only when the data format or distribution changes |
| Servers without internet | Wheel package with all dependencies and BERT in a local cache |
