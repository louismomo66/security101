const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, LevelFormat, Header, Footer, PageNumber, TabStopType,
} = require('docx');
const fs = require('fs');

const CW = 9026;
const ACCENT = '1F4E79';
const GREY = '595959';

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 150, line: 288 },
  children: [new TextRun({ text, size: o.size ?? 21, bold: o.bold, color: o.color })],
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 340, after: 150 },
  children: [new TextRun({ text, bold: true, size: 28, color: ACCENT })],
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 240, after: 110 },
  children: [new TextRun({ text, bold: true, size: 22, color: ACCENT })],
});

const B = (text) => new Paragraph({
  numbering: { reference: 'b', level: 0 },
  spacing: { after: 90, line: 288 },
  children: [new TextRun({ text, size: 21 })],
});

const cell = (t, { bold = false, shade = null, w } = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: 'auto' } : undefined,
  margins: { top: 90, bottom: 90, left: 120, right: 120 },
  children: [new Paragraph({
    spacing: { after: 0, line: 264 },
    children: [new TextRun({ text: t, bold, size: 19,
                             color: bold && shade === ACCENT ? 'FFFFFF' : undefined })],
  })],
});

const table = (heads, rows, w) => new Table({
  width: { size: CW, type: WidthType.DXA },
  columnWidths: w,
  rows: [
    new TableRow({ tableHeader: true, cantSplit: true,
      children: heads.map((h, i) => cell(h, { bold: true, shade: ACCENT, w: w[i] })) }),
    ...rows.map((r, ri) => new TableRow({ cantSplit: true,
      children: r.map((c, i) => cell(c, { w: w[i], shade: ri % 2 ? 'F2F5F9' : null })) })),
  ],
});

const Note = (title, lines) => new Table({
  width: { size: CW, type: WidthType.DXA },
  columnWidths: [CW],
  rows: [new TableRow({ children: [new TableCell({
    width: { size: CW, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: 'FDF3E7', color: 'auto' },
    margins: { top: 150, bottom: 150, left: 190, right: 190 },
    borders: {
      left: { style: BorderStyle.SINGLE, size: 18, color: 'D97706' },
      top: { style: BorderStyle.SINGLE, size: 2, color: 'E8D5BC' },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: 'E8D5BC' },
      right: { style: BorderStyle.SINGLE, size: 2, color: 'E8D5BC' },
    },
    children: [
      new Paragraph({ spacing: { after: 70 },
        children: [new TextRun({ text: title, bold: true, size: 21, color: '92400E' })] }),
      ...lines.map(t => new Paragraph({ spacing: { after: 50, line: 276 },
        children: [new TextRun({ text: t, size: 20 })] })),
    ],
  })] })],
});

const Sp = (h = 140) => new Paragraph({ spacing: { after: h }, children: [] });

const c = [];

/* ── Cover ─────────────────────────────────────────────────────────── */
c.push(
  new Paragraph({ spacing: { before: 2000, after: 0 }, children: [
    new TextRun({ text: 'CONCEPT NOTE', bold: true, size: 21, color: GREY,
                  characterSpacing: 60 })] }),
  new Paragraph({ spacing: { before: 180, after: 60 }, children: [
    new TextRun({ text: 'SENTINEL', bold: true, size: 60, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 260 }, children: [
    new TextRun({ text: 'Camera analytics for crime, traffic and road safety',
                  size: 28, color: '333333' })] }),
  new Paragraph({
    border: { top: { style: BorderStyle.SINGLE, size: 12, color: ACCENT } },
    spacing: { after: 260 }, children: [] }),
  P('Uganda has thousands of CCTV cameras. Almost nobody watches them live. '
    + 'Sentinel watches continuously and gives officers a short list of moments '
    + 'worth their attention.', { size: 23 }),
  Sp(700),
  new Paragraph({ spacing: { after: 90 }, children: [
    new TextRun({ text: 'Prepared for: ', bold: true, size: 21 }),
    new TextRun({ text: 'Uganda Police Force and partner ministries', size: 21 })] }),
  new Paragraph({ spacing: { after: 90 }, children: [
    new TextRun({ text: 'Status: ', bold: true, size: 21 }),
    new TextRun({ text: 'Research prototype, not ready for deployment', size: 21 })] }),
  new Paragraph({ spacing: { after: 90 }, children: [
    new TextRun({ text: 'Date: ', bold: true, size: 21 }),
    new TextRun({ text: 'August 2026', size: 21 })] }),
  Sp(400),
  Note('In one line', [
    'Sentinel finds moments in CCTV footage that an officer should look at. For crime '
    + 'it never decides guilt or identifies anyone. For traffic offences it can issue '
    + 'the penalty directly, the way Uganda’s Express Penalty System already does.',
  ]),
  new Paragraph({ children: [new PageBreak()] }),
);

/* ── 1. Problem ────────────────────────────────────────────────────── */
c.push(
  H1('1. The problem'),
  P('Uganda has installed cameras across all 19 policing divisions of Kampala '
    + 'Metropolitan and in towns including Masaka, Mbarara, Gulu, Arua and Jinja. '
    + 'The national target is around 5,552 cameras, and a third phase is proposed.'),
  P('The cameras work. The problem is that footage has to be watched, and nobody '
    + 'can watch thousands of streams at once. So cameras are used mostly after a '
    + 'crime is reported, to reconstruct what happened. Prevention and fast response '
    + 'are largely out of reach.'),
  P('Four things go wrong on the same streets, handled today by different units '
    + 'with different data:'),
  B('Phones and bags are snatched. The act takes under a second, the thief is often '
    + 'on a boda boda, and many victims never report it because they expect nothing '
    + 'to come of it.'),
  B('People are robbed, sometimes at knife or gun point.'),
  B('Drivers run red lights and ride the wrong way. Road deaths reached 5,144 in '
    + '2024, up about 81% in ten years.'),
  B('Traffic sits still because signals run on fixed timers that cannot respond to '
    + 'actual queues.'),
  P('All four are visible on cameras that already exist. The footage is simply going '
    + 'unwatched.'),
  Note('The key point', [
    'The gap is not more cameras. It is the capacity to review what they record.',
    'Success means an officer can safely ignore most footage, and trust that the '
    + 'small remainder is worth opening.',
  ]),
);

/* ── 2. What it does ───────────────────────────────────────────────── */
c.push(
  H1('2. What Sentinel does'),
  P('One camera feed, five functions. They share the same underlying analysis and '
    + 'differ in the rules applied on top.'),
  Sp(60),
  table(
    ['Function', 'What it finds', 'Who acts'],
    [
      ['Street theft', 'Phone and bag snatching, pickpocketing',
        'Officer reviews a clip'],
      ['Robbery and assault', 'Weapons, striking, restraint',
        'Response and investigation'],
      ['Traffic offences', 'Red lights, wrong way, illegal turns',
        'Penalty issued automatically; owner can appeal'],
      ['Road accidents', 'Collisions, a person left down, a vehicle leaving',
        'Emergency dispatch'],
      ['Traffic control', 'Queue length and flow per lane',
        'Signal timing: no person involved'],
    ],
    [1900, 4300, 2826],
  ),
  Sp(180),
  P('Traffic control sits apart from the rest. It counts vehicles, keeps no record '
    + 'about any individual, and leads to no penalty. Everything above it concerns '
    + 'identifiable people and is governed more strictly.'),
  H2('What it will not do'),
  B('No face recognition. No identifying who anyone is.'),
  B('No automatic penalties for crime. An officer decides.'),
  B('No predicting who will commit an offence.'),
  B('No claim to know intent. The system sees movement, not motive.'),
  new Paragraph({ children: [new PageBreak()] }),
);

/* ── 3. How it works ───────────────────────────────────────────────── */
c.push(
  H1('3. How it works'),
  P('Cheap checks run first and throw most footage away. Expensive analysis only '
    + 'runs on what survives. The last step is always a person.'),
  Sp(60),
  table(
    ['Step', 'What happens'],
    [
      ['1. Filter', 'Skip empty scenes'],
      ['2. Detect', 'Find people, vehicles and weapons in each frame'],
      ['3. Track', 'Follow vehicles over time: speed, lane, queue, collisions'],
      ['4. Read movement', 'Reduce each person to a stick figure and read how two '
        + 'people move together, which is what a snatch looks like'],
      ['5. Officer', 'Confirm, dismiss, or escalate'],
    ],
    [1900, 7126],
  ),
  Sp(180),
  H2('Why stick figures'),
  P('For the crime functions the system reduces each person to 17 body points and '
    + 'reads how those points move. This has three consequences that matter.'),
  B('It suits the problem. A snatch is two bodies coming together fast and '
    + 'separating fast. That is a movement pattern, and movement is what body points '
    + 'capture.'),
  B('It cannot identify anyone. The model never sees faces, skin or clothing, only '
    + 'joint positions. Privacy is built into the design, not promised on top of it.'),
  B('It runs on cheap hardware at the camera, so footage need not be shipped to a '
    + 'central server.'),
  H2('Traffic uses a simpler method'),
  P('Traffic offences, accidents and signal timing do not use stick figures at all. '
    + 'They track vehicles. A red-light violation is a vehicle crossing a line while '
    + 'the signal is red: geometric, explainable, and far more reliable than reading '
    + 'human intentions. This part of the system is considerably more mature.'),
  H2('How it is built'),
  B('Every camera is registered once with a fixed latitude and longitude. Every '
    + 'alert it raises inherits that position automatically, without depending on a '
    + 'street address or a description someone has to type in.'),
  B('Open components, so any alert can be explained to a magistrate.'),
  B('Measured against footage labelled by hand, never against impressions.'),
  B('Tuned and run by Ugandan staff on Ugandan data.'),
  B('Attaches to existing cameras; no vendor lock-in.'),
);

/* ── 4. Officers and penalties ─────────────────────────────────────── */
c.push(
  new Paragraph({ children: [new PageBreak()] }),
  H1('4. Officers and penalties'),
  P('When the system flags something, it does not simply raise an alarm. It builds a '
    + 'package an officer can act on:'),
  B('The clip, with a few seconds either side.'),
  B('Camera ID, GPS coordinates, date and time.'),
  B('Why it fired, and how confident it was.'),
  B('A tamper-evident record of the original footage.'),
  B('The officer’s decision and who made it.'),
  P('Dismissals are logged as fully as confirmations. A system whose mistakes '
    + 'disappear quietly cannot be audited.'),
  H2('Traffic offences: Sentinel issues the penalty'),
  P('A red-light violation, a wrong-way turn, an illegal turn: these are facts about '
    + 'a vehicle, not judgements about a person. Uganda already treats them this way: '
    + 'the Automated Express Penalty System issues fines from camera evidence and '
    + 'notifies the registered owner by SMS or email, without an officer reviewing '
    + 'each case first. It launched in April 2025, was paused about a month in June '
    + '2025 to fix gaps, then resumed in pilot corridors.'),
  P('Sentinel is built to do the same job at the point of detection: capture the '
    + 'violation, the plate, the camera’s GPS coordinates, and the evidence clip; '
    + 'assemble the penalty notice; and '
    + 'issue it through the existing Express Penalty channel, rather than waiting for '
    + 'an officer to first spot the case in a review queue. This is what makes traffic '
    + 'different from crime in this note: the offence is mechanical to establish, '
    + 'the liability sits with a registered vehicle rather than an unidentified '
    + 'person, and the owner keeps the same right to appeal that the existing system '
    + 'already gives them.'),
  P('An officer still enters the loop, but after issuance rather than before it: '
    + 'appealed cases route to a person, and every notice is logged with the evidence '
    + 'behind it. Nothing about this changes how crime alerts are handled; those '
    + 'stay under the rule below.'),
  H2('Crime: the officer decides'),
  P('Snatching, robbery and assault are different, and automatic penalties are not '
    + 'proposed. Three reasons, any one of which is enough.'),
  P('The accuracy is not there yet. Section 6 gives the numbers.'),
  P('More importantly, the system cannot see what the offence requires. It sees a '
    + 'hand entering a pocket, or a grab followed by two people separating. It does '
    + 'not see who owns the phone or whether it was given freely. That information '
    + 'is not in the picture, and no better model will find it. A parent taking a '
    + 'phone from a child looks the same as a thief taking one from a stranger.'),
  P('And a traffic fine attaches to a registered vehicle. A theft allegation attaches '
    + 'to a person, and carries consequences an SMS cannot fairly deliver.'),
  H2('Accidents are about help, not blame'),
  P('Collision detection exists to get an ambulance moving and to preserve the '
    + 'footage. Because the coordinates come from the camera, not a caller’s '
    + 'description of where they are, dispatch can route to the exact point rather '
    + 'than the nearest known landmark. Who was at fault is for an investigator to '
    + 'decide.'),
);

/* ── 5. Uganda ─────────────────────────────────────────────────────── */
c.push(
  new Paragraph({ children: [new PageBreak()] }),
  H1('5. Fit for Uganda'),
  H2('It uses cameras you already have'),
  P('Sentinel is software over existing cameras. Most of the cost of a camera '
    + 'programme is hardware and installation; analytics is a small fraction. '
    + 'Improving what installed cameras yield is better value than installing more '
    + 'that nobody can watch.'),
  H2('Camera quality is the real limit'),
  P('We tested the system on real Ugandan footage, including a police compilation of '
    + 'street crimes. The findings were blunt:'),
  Sp(60),
  table(
    ['What we found', 'What it means'],
    [
      ['Footage was 626 x 360; people about 20 pixels tall',
        'Too small for the system to read body movement'],
      ['At one busy market scene it found no people at all',
        'The incident is invisible, not merely difficult'],
      ['Enlarging the image 2x and 3x changed nothing',
        'The detail is not in the pixels; better software cannot recover it'],
      ['Night footage performed much worse',
        'Lighting is part of the camera specification'],
    ],
    [4000, 5026],
  ),
  Sp(180),
  Note('The most important finding in this note', [
    'Camera resolution, not software quality, is the limit on the crime functions.',
    'A few well-specified cameras at known hotspots will beat many low-resolution '
    + 'ones. Camera choice and siting should be treated as part of the system design.',
  ]),
  H2('Local conditions'),
  B('Boda bodas are involved in many snatches, a strong signal the system can use, '
    + 'and also the main source of false weapon alerts.'),
  B('Markets and taxi parks are the highest-risk places and the hardest to analyse, '
    + 'because people block each other from view.'),
  B('Sending video continuously is too expensive, so analysis happens at the camera '
    + 'and only flagged clips are sent.'),
  B('All processing can stay inside Uganda. Nothing needs to leave the country.'),
);

/* ── 6. Honest status ──────────────────────────────────────────────── */
c.push(
  new Paragraph({ children: [new PageBreak()] }),
  H1('6. Where it actually stands'),
  P('This section is here because a proposal without it would be misleading. '
    + 'Sentinel is a prototype. The functions are not equally ready.'),
  Sp(60),
  table(
    ['Function', 'Readiness'],
    [
      ['Traffic control', 'Most ready. Vehicle counting is reliable and signal timing '
        + 'has been simulated on Kampala junctions'],
      ['Traffic offences', 'Detection rules built, not yet tested against labelled '
        + 'Ugandan violations. Automatic issuance additionally needs a legal '
        + 'integration agreement with the Express Penalty System before it can go '
        + 'live: not yet started'],
      ['Accidents', 'Rules built, not yet scored against confirmed collisions'],
      ['Weapons', 'Withdrawn and being rebuilt, see below'],
      ['Snatching', 'Least ready. Works on clear footage, fails on poor footage'],
    ],
    [2400, 6626],
  ),
  Sp(180),
  H2('Snatch detection, measured'),
  P('On a clear 720p clip the system detected a motorcycle snatch. On a snatch '
    + 'through a car window it failed, because it could only make out one of the two '
    + 'people. On the 626 x 360 police footage it could not see two people in any '
    + 'frame. False alarms currently run at roughly three every fifteen seconds, far '
    + 'too high to put in front of an operator.'),
  H2('Two failures worth reporting'),
  P('A weapon detector trained in August 2026 fired on 7% of frames of footage that '
    + 'contained no weapons at all. Checking every false alarm, most were motorcycle '
    + 'fuel tanks and handlebars, plus a car at night and an animal. The cause was the '
    + 'training pictures: guns in them filled about 43% of the frame, while a real gun '
    + 'on a street camera fills about 0.4%. The model had learned size, not shape. It '
    + 'has been switched off and is being rebuilt on properly scaled images.'),
  P('A second component used a language model to double-check uncertain cases. It '
    + 'agreed with whatever the question suggested. Asked whether two people were '
    + 'present, it said yes about a woman sitting alone in a car. It now asks every '
    + 'question both ways and discards contradictory answers, and stays switched off '
    + 'until it earns its place.'),
  H2('What will not improve'),
  B('The system sees movement and objects, never intent or ownership.'),
  B('It cannot identify people, by design.'),
  B('It cannot see what the camera did not capture.'),
  B('It will always produce some false alarms. The workflow assumes this.'),
);

/* ── 7. Law ────────────────────────────────────────────────────────── */
c.push(
  new Paragraph({ children: [new PageBreak()] }),
  H1('7. Law and governance'),
  P('Uganda was the first country in East Africa to protect data privacy in dedicated '
    + 'law. The relevant instruments:'),
  Sp(60),
  table(
    ['Instrument', 'Why it matters'],
    [
      ['Constitution, Article 27', 'Guarantees the right to privacy'],
      ['Data Protection and Privacy Act 2019',
        'Governs handling of personal data. Video of identifiable people is personal '
        + 'data. Section 10 bars processing that infringes privacy'],
      ['Data Protection Regulations 2021',
        'Requires registration with the supervisory authority'],
      ['Personal Data Protection Office (NITA-U)',
        'The regulator this system would answer to'],
      ['Express Penalty System', 'The existing route for camera-derived traffic fines'],
    ],
    [2600, 6426],
  ),
  Sp(180),
  H2('The gap'),
  P('The 2019 Act was passed without strong enforcement machinery, and Uganda has no '
    + 'rule specifically covering automated analysis of public video: no required '
    + 'accuracy standard, no mandatory impact assessment, no limit on reusing the data '
    + 'for other purposes. A system like this could be deployed with less scrutiny '
    + 'than it deserves. That is a reason to adopt controls voluntarily, not a reason '
    + 'to move quietly.'),
  H2('What should be in place before a pilot'),
  B('A Data Protection Impact Assessment lodged with the regulator.'),
  B('Registration as a data collector and processor.'),
  B('A published retention schedule, enforced automatically rather than by policy.'),
  B('A written limit tying the system to the stated purposes only.'),
  B('Access logs naming every officer who views a clip, open to an independent body.'),
  B('A public notice: where cameras are, what is detected, how to complain.'),
  B('A written ban on face recognition and links to identity registers.'),
  H2('Risks to manage'),
  B('Scope creep: a system justified for theft being used for other monitoring.'),
  B('Evidence being challenged in court, which is why chain of custody is built in.'),
  B('Unequal error rates across groups. Detection models are documented to vary by '
    + 'skin tone and body size. Error rates should be measured separately and '
    + 'published.'),
  B('Discouraging lawful assembly. The system deliberately excludes crowd and '
    + 'gathering analysis.'),
);

/* ── 8. Power ──────────────────────────────────────────────────────── */
c.push(
  new Paragraph({ children: [new PageBreak()] }),
  H1('8. Power supply'),
  P('Uganda’s grid has both scheduled load shedding and unplanned outages. In 2026 '
    + 'there were national blackouts on 12 April and 5 June. Reinforcement is under '
    + 'way, but any system installed today must expect interrupted power for its '
    + 'whole life.'),
  Note('Design position', [
    'Power cuts are treated as normal operation, not as a fault. A system that needs '
    + 'mains power to work will be off during night-time outages, exactly when '
    + 'street crime risk is highest.',
  ]),
  H2('How it copes'),
  B('Analysis happens at the camera, so a site keeps working when the network drops.'),
  B('Flagged clips are stored locally and sent when the connection returns, carrying '
    + 'their original timestamp.'),
  B('Each site runs on a battery sized for 12 to 24 hours, recharged by solar. The '
    + 'camera and computer together draw 15 to 25 watts.'),
  B('Changeover is automatic. Sites are unattended, and any manual step will '
    + 'eventually not happen.'),
  H2('Running low'),
  P('As the battery drains the site sheds work in order rather than dying suddenly: '
    + 'full analysis, then detection only, then recording only, then a clean shutdown '
    + 'that protects stored evidence.'),
  P('Power state is reported alongside the alerts. This matters more than it sounds: '
    + 'an operator must be able to tell "no crime happened" from "the camera was off". '
    + 'Gaps in coverage are logged explicitly.'),
);

/* ── 9. Plan ───────────────────────────────────────────────────────── */
c.push(
  new Paragraph({ children: [new PageBreak()] }),
  H1('9. How to proceed'),
  P('Each phase starts only when the previous one has met its condition. Timelines '
    + 'are indicative; the conditions are not.'),
  Sp(60),
  table(
    ['Phase', 'Work', 'Condition to move on'],
    [
      ['1. Data', 'Label Ugandan footage for each function; agree camera '
        + 'specifications', 'A labelled test set for every function claimed ready'],
      ['2. Traffic first', 'Signal timing and traffic offences to pilot; automatic '
        + 'penalty issuance agreed with the Express Penalty System operator',
        'Offences scored against labelled violations at a false-alarm rate low '
        + 'enough to fine on; issuance agreement signed; timing improves delay '
        + 'without starving any approach'],
      ['3. Accidents', 'Run alongside existing reporting without relying on it',
        'Detects confirmed collisions at a false-alarm rate dispatchers accept'],
      ['4. Crime', 'Weapons and snatching with officer review; impact assessment '
        + 'lodged', 'Weapon model passes the false-alarm test; prosecutors accept the '
        + 'evidence packages'],
      ['5. Wider use', 'Extend deployment', 'Independent evaluation; error rates '
        + 'published by function and by group'],
    ],
    [1700, 3700, 3626],
  ),
  Sp(180),
  Note('Sequence matters', [
    'Traffic functions rest on vehicle tracking, which is mature. Snatch detection is '
    + 'an open research problem with measured failures.',
    'Treating all five as one deliverable slows the programme to the pace of the '
    + 'hardest part, and risks the sound parts being judged by the weakest.',
  ]),
  H2('How to tell if it is working'),
  B('Officer hours spent searching footage per case solved.'),
  B('Share of flagged clips that turn out to be real.'),
  B('False alarms per camera per day: the number that decides whether operators keep '
    + 'trusting it.'),
  B('Time between an incident and an officer knowing.'),
  H2('When to stop'),
  B('False alarms so frequent that operators ignore alerts.'),
  B('Error rates that differ by group and do not close with better data.'),
  B('Use of the system beyond its authorised purpose.'),
  B('Cameras that cannot be specified well enough at the sites that matter.'),
);

/* ── 10. Close ─────────────────────────────────────────────────────── */
c.push(
  new Paragraph({ children: [new PageBreak()] }),
  H1('10. Conclusion'),
  P('The expensive part is already done. Cameras are installed across Kampala and '
    + 'beyond, and a third phase is under consideration. What is missing is the '
    + 'capacity to watch what they record, which is why they are used mainly to look '
    + 'backwards.'),
  P('Sentinel adds automated triage to cameras that already exist, across five '
    + 'related functions, and hands officers reviewable evidence while leaving every '
    + 'consequential decision with them. It reads body movement rather than faces, so '
    + 'its privacy properties are structural. It is designed for interrupted power and '
    + 'patchy connectivity because those are the real conditions.'),
  P('It is not ready. Section 6 is honest about that, and the binding limit is camera '
    + 'resolution rather than software, a finding that should shape procurement as '
    + 'much as engineering. The sensible path is staged, starting with the traffic '
    + 'functions, which are closest to working.'),
  H2('What is asked now'),
  B('Access to representative footage under an appropriate agreement.'),
  B('Guidance from officers on how review would fit their working day.'),
  B('An early conversation with the Personal Data Protection Office.'),
  Sp(200),
  new Paragraph({
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: 'C9D3E0' } },
    spacing: { before: 120, after: 160 }, children: [] }),
  P('Sources: Uganda Police Force and Parliament of Uganda (national CCTV); Ministry '
    + 'of Works and Transport (Express Penalty System); Data Protection and Privacy '
    + 'Act 2019 and Regulations 2021; NITA-U; Uganda Electricity Distribution Company '
    + 'and Ministry of Energy; 2024 Police Annual Crime Report.',
    { size: 18, color: GREY }),
);

const doc = new Document({
  creator: 'Sentinel Project',
  title: 'Sentinel Concept Note',
  description: 'Camera analytics for crime, traffic and road safety',
  numbering: { config: [{ reference: 'b', levels: [
    { level: 0, format: LevelFormat.BULLET, text: '•',
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 440, hanging: 250 } } } }] }] },
  styles: { default: { document: { run: { font: 'Calibri', size: 21 } } } },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      spacing: { after: 0 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'C9D3E0' } },
      tabStops: [{ type: TabStopType.RIGHT, position: CW }],
      children: [
        new TextRun({ text: 'Sentinel: Concept Note', size: 17, color: GREY }),
        new TextRun({ text: '\t', size: 17 }),
        new TextRun({ text: 'For official consideration', size: 17, color: GREY }),
      ] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ['Page ', PageNumber.CURRENT, ' of ',
                                          PageNumber.TOTAL_PAGES],
                               size: 17, color: GREY })] })] }) },
    children: c,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync('Sentinel_Concept_Note.docx', b);
  console.log('written', b.length, 'bytes');
});
