from mgz import settings
from mgz.ds.sentence_datasets.gpt_input_augments import \
    InputAugment
from mgz.ds.sentence_datasets.sentence_datasets import \
    strings_to_padded_id_tensor_w_mask
from mgz.version_control import ModelNode, ModelDatabase

model_node: ModelNode = ModelDatabase.mistral_openchat_quantized(
    "AdaptLLM/law-chat")
print(model_node.tokenizer.special_tokens_map)

email = """Message-ID: <1584518.1075840160651.JavaMail.evans@thyme>
Date: Wed, 23 May 2001 05:52:05 -0700 (PDT)
From: kevinscott@onlinemailbox.net
To: jeff.skilling@enron.com
Subject: Lockyer Fires Earthy Attack at Energy Exec
Cc: sherri.sera@enron.com
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: quoted-printable
Bcc: sherri.sera@enron.com
X-From: Kevin Scott <kevinscott@onlinemailbox.net>@ENRON
X-To: Skilling, Jeff </o=ENRON/ou=NA/cn=Recipients/cn=JSKILLIN>
X-cc: Sera, Sherri </o=ENRON/ou=NA/cn=Recipients/cn=SREINAR>
X-bcc: 
X-Folder: \jskillin\Inbox
X-Origin: SKILLING-J
X-FileName: jskillin.pst


Jeff
=20
FYI - Strong and colorful words  from a powerful man.
=20
Given Lockyer's abilities,  position and and ambition, I would advise build=
ing bridges and mending fences  while this is still at the taunting stage. =
 He wants your  attention.  Knowing him, I'd say that a direct and friendly=
  call from you or Ken today, followed by a meeting would go a long  way.
=20
Kevin
213-926-2626
=20
=20
 [IMAGE] [IMAGE] [IMAGE] [IMAGE]     News  Politics  Entertainment  music ,=
 movies , art , TV , restaurants  [IMAGE] Business  Travel  Marketplace  jo=
bs , homes , cars , rentals , classifieds  [IMAGE] Sports  Commentary  Shop=
ping     [IMAGE] [IMAGE]  [IMAGE] [IMAGE]    [IMAGE]  [IMAGE]    California=
 [IMAGE]     [IMAGE]  TOP STORIES * Bishop Asked to  Quit for Defying Churc=
h    * Wide-Ranging  Debate Reveals Much Accord    * Limit on New Sea  Wall=
s Urged      MORE [IMAGE] [IMAGE]   [IMAGE]  STORIES BY DATE FOR THIS  SECT=
ION  5/23  |  5/22  | 5/21  | 5/20  | 5/19  | 5/18  | 5/17     [IMAGE]     =
  DAILY SECTIONS   Front Page  "A"  Section  California  [IMAGE] Business  =
Sports  Calendar  [IMAGE] So. Cal. Living  Editorials,Letters,  Op/Ed     W=
EEKLY SECTIONS   Health  Food   [IMAGE] Tech Times   [IMAGE] Highway 1     =
SUNDAY SECTIONS   Book Review  Opinion  Real Estate   [IMAGE] Calendar  Mag=
azine  Travel   [IMAGE] TV Times  Work  Place   [IMAGE]  [IMAGE] [IMAGE]  [=
IMAGE] [IMAGE]    [IMAGE] Marketplace   Find a home , car ,  rental , job ,=
  pet , merchandise , boat, plane or RV , classifieds   Place an  Ad  [IMAG=
E] [IMAGE] [IMAGE]  [IMAGE] [IMAGE]    [IMAGE] L.A. Times Subscription  Ser=
vices    Subscribe , Change of Address , Vacation Stops , Suspend Delivery =
, College Discount , Gift Subscriptions , Mail Subscriptions , FAQ    [IMAG=
E] [IMAGE] [IMAGE]  [IMAGE] [IMAGE]    [IMAGE] Print Ads  from the Newspape=
r    See  this week's ads [IMAGE] [IMAGE]      [IMAGE] Print Edition , Oran=
ge  County , Valley , Ventura County , National ,  Community  Papers  [IMAG=
E]    [IMAGE] [IMAGE] [IMAGE]     Books  Columnists  Crossword  Education  =
Food  Health  Highway    Horoscope     Lottery  Magazine  Obituaries  Readi=
ng by    Real Estate  Religion  Science  So.Cal. Living     Special  Report=
s  Sunday Opinion  Tech  Times  Times Poll  Traffic  Weather  Workplace  SI=
TE  MAP       [IMAGE]   [IMAGE] [IMAGE]    SHOP 'TIL YOUR LAPTOP DROPS [IMA=
GE] [IMAGE]  [IMAGE] [IMAGE]     [IMAGE]   Shopping [IMAGE] Search     Prod=
ucts Stores  [IMAGE]      [IMAGE]    [IMAGE]  [IMAGE]     [IMAGE]  [IMAGE] =
=09[IMAGE]=09[IMAGE] Wednesday, May 23, 2001 | [IMAGE]Print this story   [I=
MAGE]  Lockyer Fires Earthy Attack at Energy Exec     By JENIFER WARREN, Ti=
mes Staff  Writer        SACRAMENTO--In a  dramatic escalation of energy cr=
isis rhetoric, California Atty. Gen. Bill  Lockyer this week suggested the =
chairman of a Houston-based power company  should be locked in a prison cel=
l with an amorous, tattooed inmate named  Spike.       Lockyer, who is inve=
stigating  whether energy firms have manipulated prices on the wholesale el=
ectricity  market, made the comment in an interview with the Wall Street Jo=
urnal that  appeared Tuesday.       "I would love to  personally escort [En=
ron Corp. Chairman Kenneth] Lay to an 8-by-10 cell  that he could share wit=
h a tattooed dude who says, 'Hi my name is Spike,  honey,' " Lockyer said. =
      Enron spokesman  Mark Palmer called the comment "counterproductive rh=
etoric" that "does not  merit a response."       But other industry  repres=
entatives denounced the remark as "outrageous," especially because  neither=
 Lockyer's office nor any investigative panel has filed charges  against En=
ron or other companies.       "You'd  expect that the state's chief legal c=
ounsel would file charges first and  make public statements second," said G=
ary Ackerman of the Western Power  Trading Forum, an association of energy =
producers and traders. "We're very  disappointed with his choice of words, =
which don't exactly fit the profile  of his office."       In an interview =
Tuesday,  Lockyer said he decided to "ratchet up" the commentary to "put [e=
nergy  companies] on notice" that "we are not afraid of them and have the w=
ill to  prosecute."       "What I'm trying to do is  let these economic buc=
caneers understand that if we catch them, they're  going to be prosecuted,"=
 Lockyer said. "Just because they're  multimillionaires and run big corpora=
tions, it doesn't provide them with  immunity."       The attorney general =
is  investigating whether power company officials tried to maximize profits=
  through illegal manipulation of prices on the wholesale energy market.  S=
everal panels, including a state Senate committee and the California  Publi=
c Utilities Commission, are conducting similar probes.        On Tuesday, L=
ockyer announced that three  power companies have agreed to turn over docum=
ents subpoenaed months ago  by his investigators. The attorney general went=
 to court to obtain the  documents after the companies failed to meet a Mar=
ch 19 deadline to hand  them over.       Lockyer said the forthcoming  docu=
ments would help his office as it sifts through mountains of evidence  in s=
earch of possible violations of antitrust or unfair business practice  laws=
.       "Evidence is accumulating that  certainly infers illegal activity,"=
 Lockyer said. "But we need to make  sure it's compelling and clear enough =
that you can convince a jury."        Lockyer said he singled out Enron's  =
chairman because the Houston company is the world's largest energy trader. =
       At least one observer found Lockyer's  comments refreshingly candid.=
 Harry Snyder, a senior advocate of Consumers  Union, said, "Let Lockyer be=
 Lockyer."  * * *      Times staff writer Dan  Morain contributed to this s=
tory.    Search  the archives of the Los Angeles Times for similar stories =
about:  Bill  Lockyer , Enron  Corp , Kenneth  L Lay , Utilitiy  Rates , En=
ergy  - California , Utilities  - California , Electricity . You  will not =
be charged to look for stories, only to retrieve one.  =09
    News  Politics  Entertainment  music , movies , art , TV , restaurants =
 [IMAGE] Business  Travel  Marketplace  jobs , homes , cars , rentals , cla=
ssifieds  [IMAGE] Sports  Commentary  Shopping     [IMAGE] =09[IMAGE]=09  G=
et Copyright Clearance   Copyright 2001 Los Angeles  Times  Click for permi=
ssion to reprint  (PRC# 1.528.2001_000043205)      =09


   [IMAGE]        =09"""
tag = 'all documents and communications between enron employees discussing government inquiries and investigations into enron'
prompt = f"According to this e-mail: \n{email}\n Does the tag \"{tag}\" apply to the e-mail?\n"
# augmented = [tag_question_augment(email, tag)]
augmented = [InputAugment.prompt_adapt(prompt, system_context="")]
src_ids, src_mask = strings_to_padded_id_tensor_w_mask(augmented,
                                                       model_node.tokenizer,
                                                       4000,
                                                       settings.DEVICE)
result = model_node.model.generate(src_ids=src_ids, src_mask=src_mask,
                                   max_new_tokens=100)
print('src_ids', src_ids.shape)
print('result', result.shape)
answer = model_node.tokenizer.batch_decode(result, skip_special_tokens=True)
for ans in answer[0].split('GPT4 Correct Assistant:'):
    print('---------')
    print(ans)
