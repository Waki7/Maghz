from __future__ import annotations
from __future__ import annotations
from __future__ import annotations

import json
import unittest

from transformers import LlamaTokenizer

import mgz.settings as settings
from mgz.ds.sentence_datasets.seq_to_sequence_datasets.aus_legal_case_reports import SampleType
from mgz.ds.sentence_datasets.responsivenes_datasets.enron_emails import EnronEmailsTagQA
from mgz.model_running.run_ops import tagging_embedding_controller, \
    embedding_controller_from_texts, summarize_controller_from_texts
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.models.nlp.led import LEDForConditionalGeneration
from mgz.models.nlp.mistral import MistralForCausalLM
from mgz.models.nlp.mistral_hug import MistralForCausalLMHug
from mgz.typing import *
from mgz.version_control import ModelNode


def importing_sample_data(tokenizer: LlamaTokenizer) -> List[
    Dict[str, Union[int, str, NDArrayT]]]:
    ds = EnronEmailsTagQA(tokenizer=tokenizer,
                          max_src_len=4096).load_validation_data()
    ds_samples = ds.data[0:20]
    ds_samples.insert(0, None)  # to make index same as ID

    return [
        {
            "id": 1,
            "title": ds_samples[1][SampleType.NAME],
            "source_text": ds_samples[1][SampleType.FULL_AS_STRING],
            "date": ds_samples[1][SampleType.KEY],
            "catchphrases": ds_samples[1][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/12/chicken-vesuvio-document.html",
            "summary": "This is a summary of a recent meeting between a Houston business leader and a Saudi business leader.  The meeting was held on a Monday and we were hard at work on a different proposal designed to speed up or progress to the top ranks of the world's best business schools.  \n\nThe proposal was developed by the Council of Overseers of the Jones School. The Council was composed of about 36 businessleaders, mostly from Houston but also includes individuals from as far awayas Saudi Arabia. The group consists of about 35 businessleaders from Houston and also includes individual from as long awayasSaudi Arabia.\n\nWe had a good meeting and discussed the matter of whether or not you would join the Council. The council was chaired by Dennis Hendrix, former CEO of PanEnergy. The charter of the group consisted of about 37 businessleaders. This group consists by about 36businessleaders, most from Houston, mostly with individuals from \"far awayas Arabia Arabia.\"\n\nAs we discussed the proposal, the Council was hard at working on a new proposal designed that speed up the or progress up or progression to the Top ranks of The world'sbest business schools, as well as the interests of much of the Houston business community. As you pointed out, the advice and counseloffered by the group to be extraordinarily helpful as we work on improvingthe Jones School was in our mutual interest as well.\n \nWe met about two times a year for an afternoon to discuss important policyissues facing the Business School. I have found the advice, counsel and counselofferered by Dennis and other businessleaders to be helpful as they work on the improvingtheJones School. \n    The Council of Osverseers was chaired and included individuals from Saudi Arabia but also included individuals as far as Saudi Arabia, as a new member or as a guest.\n\nOn the other hand, the council was formed by 36 business leaders from Houston who also includes an individual from Saudi Saudi Arabia as well includes individuals as long as Saudi Saudi Arabian. The members of the council included individuals who have as far afieldas Saudi Saudi Aramco. The member of the Council is chaired by a former CEO. The Charter is comprised of about 42 businessleaders mostly from Texas but also including individuals from far away as Saudi Arabian Arabia.  On the council of overseers, the group consists about 36 Businessleaders, almost from Houston.  They also includes members from as close awayas Riyadh Arabia. As we discussed, the proposal was hard-working and would speed up and progress to either the top ranking of the best business school.\n\n\n \nOn the Council's Charter, the charter of this group consists and the Council, the members of this council, and the charter The Council's charter, and their members, the Charter of the City.\n\t\nOn June 30, the City's Charter was formed to include 36 business-leaders, including individuals who had as farawayas Saudi Arabian as Saudi Aramcavian. The City's charter consisted of 36 businessleader, mostlyfrom Houston but included individuals with as far fromas SaudiSaudi Arabia, and included individual individuals from acrossas Saudi Arab Arabia. It was hard work on this different proposal.\n<a href=\"http://www.enron.com/enron/enro/en-ron/rice-edu/news/article/article-advanceers-of-the-jones-school/article_overviewers-and-adverseers-interests in the area of e-commerce.html\">The Council of oversers of Jones School</a>\">On June 20, 2000</a>, the Houston Business Council Group was formedThe CouncilFrom Houston to Saudi ArabiaFrom the Council, From the Council: to Houston, Houston Houston, Arabia Houston and Saudi Arabia to Saudi Arabia, Saudi Arabia and Yemen StateHouston College TexsLocation City Area Neighborhood Washington ) San Wyoming St Web North Austin Texas New Midwest Houston College Houston University Colorado Pennsylvania Maryland Wisconsin University Washington, Washington State Washington University Wyoming, State University Houston Washington College Washington National National University, Texas, University of Saudi Arabia ( Christian Country USA Denver Oklahoma Middle W Law Minneapolis Chicago Missouri TCU Stanford Harvard Louisiana Idaho Utah Boise Seattle Southern West Rice University Houston Business Dallas\n"
        },
        {
            "id": 2,
            "title": ds_samples[2][SampleType.NAME],
            "source_text": ds_samples[2][SampleType.FULL_AS_STRING],
            "date": ds_samples[2][SampleType.KEY],
            "catchphrases": ds_samples[2][SampleType.CATCHPHRASES],
            "url": "http://nodocuments.com/document/chicken-paprikash/",
            "summary": "May 15 conference: The deregulation conference in Tokyo, in person or by videoconference, is set for May 15, 2001.  We have no further information on this matter.  However, the possibility of a speaker from FERC for the May 15 deregulation conference is very close to our attention.  The FERC Chairman is a former FERC member who was able to secure a speaker for the conference in person.  He was unable to secure one for the June deregulation conference, so we have no information on his availability.  \n\nOn March 14, 2001, we talked about the possibility that securing a speaker would be a good idea.  In fact, we have a good chance of securing a Speaker from FRC for the May 15 conference in June.  But we have not yet decided on a suitable alternative.  So, let's move on to the next topic.  <i>>> > <i><a href=\"http://www.clearinghouse.net/detail.php?id=12787\">The Washington Post</a>, wrote to me on March 14th, 2001 about the potential of securing speakers from F.R.I.C. for the deregulation conference.  I have no more information on that subject.>On the subject of a potential speaker, we discussed the possibility on March 15, 2002.The conference was held in Tokyo on May 15.\"The May 15 event was\"The May15 conference was>The May 14 conference was... \"\nOn the May15 Conference: The May 15 Conference: \n\n\nThe June 15 deregulation event in Tokyo is set to take place in person and by video conference.\n\t and the May 14 deregulation conference: Joe, could you see if someone at FERC -- Hebert or Madden, might be available -- or suggest a suitableAlternative: Joe <a href= \"https://www2.net\">Joe</a> \"The May\" conference: <a>\n\n>\n>\n>\t> \n>>\n>\n</a>)>)>>\">\n>Joe,\">Joe, the May</a>.\n> Joe,>Joe </a>Joe>Joe -- could you seeing if someone could?....> Joe>Joe,...,>...\n \n \n</>Joseph>Joe...... Joe> Joe Joe>...Joe, he //......Joe> Joe,..\n < |\t> Joe..............\n> \n\t>Joe>,.\">.\"\n>\".\"'',\",\".\",\"''s\"''\"\">...\",\".\"'';>'' \"\"\">..\".\">,.\" he.\" --- ---.\" --...\"\u00a0\u00a0\u00a0>'';\"/>\n>'';.\",.\"...... )\ufffd\u00bb.. \"\" /// \u00bb }> //....\" ).\".......,.\".,....\".....\".. ---........\"....\"...\".\".............\".\" ---- [...] ; ----.\" .\"\u00a0\u00a0\u00a0 --.\"'';. )..... ;.\" ;........................\"... )... ;........ he.,\".\"...\"...,...\",,......\"... //.\"...\"...\"....\".\"...\". [...].\" //.\"-\" he.\" ).\"'';.\",..\".\".\"..\"...\" ---.\" ---.\".\" ---.......\"...\".\".\".\"...\".\" //..\" ).\".\" )....\".'';..\"....\".\".....\"...\".\".\"...\"..\",.\".\",........\"...\" ;.\"....\",\".\".).\" );.\".).).\" ); );.\" [...].\">.\"''. // )... )'';'';............ ;. } ___ [\u2026] ] ).....?-\".\" [...]. [\u2026]. );..). --. \u2026"
        },
        {
            "id": 3,
            "title": ds_samples[3][SampleType.NAME],
            "source_text": ds_samples[3][SampleType.FULL_AS_STRING],
            "date": ds_samples[3][SampleType.KEY],
            "catchphrases": ds_samples[3][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/02/cauliflower-and-tofu-curry-document.html",
            "summary": "Chili Cook Off at Rice University is scheduled for October 5th, at Rice's Ray Courtyard, Student Center, and Ray Courthouse.  The students taste your chili and then they vote for the best.    The Rice vs. Fresno game is held 2 hours before the game.  A zydeco band will provide the entertainment and each company will be given 4 complimentary football tickets for the 6pm game. The students also vote for their best chef.  During the game, the students taste the chili and they vote.  There will be free hotdogs, sodas, and beer, and a zydro band will be providing the entertainment.  Food was provided and eachcompany was given 4 free football tickets.  After the game the students will be awarded 4 complimentary tickets for 6pm, and the 6 pm game.\n\nThe Chili Cook Off is held at Rice Stadium and is held in 2 hours prior to the Rice vs Fresno game. There will also be free food, soda, and free football ticket for the 7pm game, and 4 complimentary baseball tickets for a 6pm show.  If the students vote for best, they will be rewarded with 4 complimentary seats for the 8pm game and the school votes will be split between the two teams.  For the students, the Chili Cook-Off is held two hours before and before the Rice-Fresno game. Admission is free, and each individual company will receive 4 complimentary ticket tickets for that 6pmgame.  In addition, the student vote will be divided between the four teams, and there will be 4 complimentary games for the 5pm game as well as 4 complimentary school tickets for both the 6 and 6pm games.  This event is scheduled to last 2 hours and is scheduled at Ray Cour Courtyard and Student Center. \n\nOn September 28, 2000, the Rice Team sent a letter to the event's organizers, inviting them to participate in the Chili Off.  On September 27, 2000 the Rice team sent a list of chefs that had worked on the Chili cook-off.  They were all awarded $100,000.  Later that month, the event was moved to the Ray Cour.  More details on the chili cook-offs can be found in the Rice website.   \n   On October 5, the school held a Chili Cook off at Rice.  It was held 2 hour before the Rice vs. Fresno football game. A zyeco band provided the entertainment, eachcompany will be provided 4 complimentary free football seats for 6 pmgame, and students voted for the most student votes.  Students were given 4 individual football tickets, and they were given the most votes.\n  The Chili Cookoff was held at the Ray Courttyard, the Student Center and the Ray Hall.  Rice University's 9th Annual Recruiter Chili Cook - Off was held on October 5.  We have no further information on this event.\n\n\n \n: The Chili cookoff is held on the same day as the Rice game.\n\n\n: The students vote on the best chili and the student votes are split between two teams, one of which will be the winner of the most students votes. The other, the winner, will be handed 4 complimentary basketball tickets for sixpm game at the 6-pm game (the student votes were split between four teams).\n\n: On October 7, the chili was split between eight teams, with four teams each receiving 4 complimentary ball tickets for their 6pm match.  Each team received 4 complimentary student votes for the six-pm match, and one of the chefs received 4 free school tickets.\n, the 7-m match, the 6th, and 6th graders received 4 individual seats.\n\t: The student votes went to the most of the student voted.\n\n\n: The Student votes were divided between four groups, with the most voted for student votes going to the 6m game. College, the University of Texas, the Yale School of Law, the College of Nursing, the Nursing Home, the Medical Center, the School of Nursing's Nursing Home and the University's Student Center's Student Activity Center.\n<i>The Student Vote</i>\n<I>The student votes on the student voting ballots are split among four teams.\n <i>Rice University's 8th Annual Recreationruiter chili Cook-off</i>, held on Oct. 5, is held before the 7th, 8th, 9th, 10th, 11th, 12th, 14th, 13th, 16th, 17th, 18th, 19th, 20th, 21st, 2023, and 22nd, respectively.\n </i> \n<li>The students taste their chili and then the student ballots are divided between two groups, one group will be each receiving a free hotdog, sodacostomy, and/or, and drink"

        },
        {
            "id": 4,
            "title": ds_samples[4][SampleType.NAME],
            "source_text": ds_samples[4][SampleType.FULL_AS_STRING],
            "date": ds_samples[4][SampleType.KEY],
            "catchphrases": ds_samples[4][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/02/cauliflower-and-tofu-curry-document.html",
            "summary": "Re: FW: Surprise!! The Secret Life of the Party is about to be revealed! <hr>Ext. 3-1586 <hr><hr>See <a href=\"https://www.clearinghouse.net/detail.php?id=1586\"> FW: Secret Life</a> for more information. </a>The Secret Life is about the Party and the Party, and the party is about...Re: F: Surprise! <a>Ext.\" 3- 1586 <a>. See <a>, <a><hr>, <i>Ext. 3-1486</a>. <a<a></a>>Ext> Ext.Ext.ExtExtExt> < Ext.Ext.Ext ExtextThe ExtensionExt Ext. Extension extEExtextExt extendsOnExt.Ext extExt?ExtAbstractExt,.Ext </Ext!Ext\ufffdExt?Ext)ExtvertisingExt>.Ext).DistExt... |.>.> Extension |. |Exthh.h? [\u2026]?> }t }Ext }>?\u2026\u00a7);\u00ab\u2022\ufffd-pdf )? [\u2026]? [\u2026] [\u2026] [\u2026]. [\u2026]? | } |?> ... [...]\"> ]| undefinedTweet ; );\u00ad... \u2013How\u00a0***]\u00bb \u00bb \u2022 ]; };\u00b7\u00c2 </>.>>\u2022\u2022\u2022]\u2022 \u2022\u2022 }\u2022.\u2022 |\u2022 ;\u2022 ]\u2022 [\u2026]\u2022 *} *\u2022>...\u2022. ;. |. }. ].\u00a0. \u2022. [...]..... [\u2026]. *. ..\"\".\".\".''''.\ufffd.?.\u2026....\"*  }. }\u2022 } } } | } * } ; }].] } \u2022 }\u00a0\u2022\u00a0 } \u00bb. \u2026'';.\ufffd`. //`''. \\';};...\u00bb. ).\u00c2. \u00bb } ] } [\u2026] } ) } ];. };.}..... //. \u00ad \u00ad... }? } }; }... }\u00b7. \u2013... \u2026. ---...\ufffd\ufffd\ufffd >>\ufffd\ufffd\ufffd. \ufffd\ufffd\ufffd?? */ \u00b7...\". \u2014 \u2014. );.*.\ufffd....\".\" ). -\". \"....\". ).. */.***. \u00b7.\".. \"..`..\".\ufffd..\ufffd.-\"vertising /> \"\"\"---\ufffd\u25a0Abstractadvertisement \u00a0 \u25cf ~ \u00d7 ---- __ \u200e ()\u200b************! })\u2014Advertisement**?) ---.??. />.vertising.advertisement.). ().\u00b7 }  ). ) ) ) } )... ) ; [\u2026] ) [\u2026] } [\u2026]... [\u2026] ; ) [\u2026] )? )\u00a0 [\u2026]  [\u2026]\u00a0 ) ?.? }? )??? ;?...? [\u2026]?\u00a0?\u00c2 }\u00c2 [\u2026]\u00c2?  ;. ; } ; ; ; ) ;? ; [\u2026] ;... ; ....... }... )......... ;... [\u2026]...?...  | [\u2026] | ) |... | ;"

        },
        {
            "id": 5,
            "title": ds_samples[5][SampleType.NAME],
            "source_text": ds_samples[5][SampleType.FULL_AS_STRING],
            "date": ds_samples[5][SampleType.KEY],
            "catchphrases": ds_samples[5][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/02/cauliflower-and-tofu-curry-document.html",
            "summary": "On June 21, 2001, a Stanford Algorithmics employee called to inform him that he was on vacation and that he needed to attend his father's 70th birthday in Montreal. The employee explained that he had to attend a father's birthday in Quebec on the 2nd in order to accommodate either day. The person who called was not available, but he did have a voice message. The voice message was from a friend of his who was on a vacation. The friend was not able to reach him because he was going on vacation. \n\nOn July 3rd, the employee informed him that his father had died on the 1st in Montreal and that the meeting had been rescheduled. The next day, the executive notified him that the Eron-Algo meeting had already been reschedule. The executive explained that the employee had been on vacation for a couple of weeks and that his wife was going to attend her fathers' 70th anniversary in Montreal on the same day. He was going off on vacation on the weekend, so he had no time to reschecdule.\n\nThe executive explained to him that they had been scheduling the meeting from July 3 to either Thursday July 5th or Friday July 6th. The meeting was scheduled for Thursday, but the employee was on his fathers 70thth birthday on the second in Montreal, so the Erons-Algos meeting was reschedigned. The individual was on the vacation and had not contacted him on Friday when he was supposed to be on vacation, so we were unable to schedule a meeting. The Executive explained that they were going to have a meeting on July 3, but that they would have to rescheddule the meeting for either day, and that they could schedule the meeting on either day if they could. The Eron meeting was set for Thursday and Friday, but we had to reschdule it to either Friday or Saturday. The day after the Eons-Alog meeting, the individual was to attend my fianc's fathers 70th birth on the first in Montreal  Let me know if you can accommodate either days. The other day, he was to have his father\u2019s 70rd birthday on either side of the meeting.  \nThe Executive explained: \nOn June 22, 2001 the executive informed him of his intention to attend the fathers' birthday on his fathers' second in- Montreal.\n </i>> </a>>\n  } | ] \u00bb'' \"\"\">\u00bb '' '''\" // \"'\"] [ ] ] \"'' \"\"I shall be..\"</ )s?\\\"> is [\u2026]...<></>  </_>\n\n\n....\n\n . ,(Updated)\n\n\t\n\n<hr>\n\n\n\nThe.</i>\n\t>\n \n\t \n     \n\n\n\n </i>,/>\n\n>\n<ol>\n\n\n\n<ul>\n> <a href=\"http://www.algorithmics.com>\\\">\n\n\"><i>P. </ul> \"Updated</a>\n </ul>)\u00a0\u00a0\u00a0\n\n\u00a0\u00a0\u00a0 inferred\n\n</i> \n>\n\n </ol> }\n </>\t <)</>>\\? );, </ </I ;.</></\u00a0 /// undefined null\t </audio * \u00b6 Transcript unspecified..............\n\n. ) </ </ )\n \"I </ ). </ \" ) \" \" </ I \"\" \"\"\"\"/> exemplary! <+-+ spurious?) ---------- [...] # tc \\ ## <@,\"/> </doc>., </ ); </statement>,. </local>\u00a0\u00a0\u00a0 </ caption </>) </summary>)</a </o></a>, </group> ) </, ), and </int </html>? </. \" ). ),.... ) ) )  ) \"... )."

        },
        {
            "id": 6,
            "title": ds_samples[6][SampleType.NAME],
            "source_text": ds_samples[6][SampleType.FULL_AS_STRING],
            "date": ds_samples[6][SampleType.KEY],
            "catchphrases": ds_samples[6][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/02/cauliflower-and-tofu-curry-document.html",
            "summary": "Hold for Mike Day/Wright and Talisman, 49c5-49, 49b5-50, 49a5-47, 49bb-50.With Mark Haedicke, keep Sylvia Sauseda and Bernadette apprised of details. This is a developing story. Stay tuned for updates.Betsy Diamond in Mike Day's office 415-781-0701is making their arrangements. Updated: Added information about the case.  Updated Date: Oct. 5, 2018 13:10 PM ETThis article was originally published on The Stanford Cardinal website.With the exception of the case of the alleged sexual assault, the case remains open. (Updated: Oct 11, 2018)BetsyDiamond in Mike's office is making their arrangementThe Stanford Cardinal is making a final decision on their own. continuesOnUpdatedWith the Stanford CardinalWrittenBackgroundThis continues:Continued ContinuedByC UpdatedSt. LouisFor the record, this is the first articleSummaryWith Mark, 49 concludesWith MarkFinally,EndNotes.UpdatedWith Mark continuesTheCurrentlySeeAbstractDocumentpic |continNothingAgain ContinueDue ClosedMoretc pt is undefined [\u2026] Transcriptpts... and;l. ) }pdf ptr)? ;\u2026 \u2026?With Mark...With Mark\u2026With Mark Mark H...\"\u2026\"\"With Mark...\"...\"...\"...\"...\"...\"...\"...\".\".\"...\".\".\".\".\".\".\"...\"\"\">''.\"\"...\"\u2026\"...\"''.''\"; \".\".;\"\"]-\"''; //\u00bb \u2026\" [...] \"\ufffd\"\u2014...\u2026.\"\"......\".\"\"\u2026\"\"...\"\"\".\"With \u2014 \u2013\ufffd\ufffd\ufffd\u00a0.. \u00b7\u00c2\"\u2026 With\u201d\".\"..\"With.\"With\"...\".\"with\").with.\"\".\"With.\" With.\"with\".\"With..\"..\"\"......\"\">With.\"\u2026\".\"\u2026\".\".\u2026\"With\u2026\"\u2026\"\u2026\"\"\u2026\".\"\u2026\"...\".\".\"\" [\u2026].\"...\".\"....\"....\".\".\" )\"-\"\"\")\")\"\";\" [...]\"\">\"\".\" \".\" ).\" [\u2026]\" \u2026\"\" \u2026\"''\" \u2014.\" \u2014\" //\" \u2013\"...\" --\"/> \"\".) --\" ;\" - -\"\u00a0\" |\"\u2026\"\").\"\ufffd.\" \u2013.\" --......\"...\"................\"....\"...\"With....\".\u2026\". [\u2026]. \u2026. [...].With...With...\"\"...\"......\"....\".\"\".WithWith...\"With...\"With\"..With.\"With.\"........\"With.\"...\".\".. \"..\u00a0. \u2014. \u2013. ).-\". \u2026\". \"\" \u00b7\" \". ;. \" . }\"\u00c2\" * ______ *\" \"\"\".)\" }.\u00c2. \"\". *. and\" ). ).\" and. //.''. |.\";..).\ufffd. )..\ufffd\ufffd\ufffd. --....\"... [...]...\"....\u00a0... \".... [\u2026]... \"... )... } ). )... ) ) )\" )...\" ) ;... ... ;.\" [...].\" \".\" \u2026......\"... |"

        },
        {
            "id": 7,
            "title": ds_samples[7][SampleType.NAME],
            "source_text": ds_samples[7][SampleType.FULL_AS_STRING],
            "date": ds_samples[7][SampleType.KEY],
            "catchphrases": ds_samples[7][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/02/cauliflower-and-tofu-curry-document.html",
            "summary": "In July 2001, Western Wholesale Activities - Gas & Power Conf. (WLAS) held in Portland, Oregon.  The conference was held in conjunction with the Western Conference of the Petroleum Manufacturers.  After a series of FERC issues and proceedings, the parties reached a settlement.  \n\nThe terms of the settlement are not publicly available, but the terms of a consent decree are pending.  We have no further information on this case.\n\n\nThe case is closed.\n\n \n\n\nThe parties are engaged in settlement negotiations.  As of July 2018, there has been no further activity on the case. \n \nThe parties have been engaged in a series discussions regarding FERC's issues and proceeding.  On July 10, CAISO's draft waiver proposal was filed.  Comments are due to CAISO this Friday.  In the meantime, the case is ongoing. \n\nOn July 25, 2001, the FERC issued a notice of non-compliance with the settlement agreement.  It stated that the parties had reached a resolution to the dispute.  However, the terms and conditions of the agreement are not available. \n\n\nThe FERC had been engaged for several months in the ongoing issues and litigation.  This case is still ongoing.\nOn July 29, 2001 the FRC held a hearing on the settlement.\n\nOn August 9, the court granted the parties' joint motion to dismiss the case, and the FCC granted the plaintiffs' motion to stay the case pending the resolution of the dispute resolution.  Thus, the matter is now closed.On September 30, 2001 FERC related issues and negotiations.\nOn October 25, the defendants filed a motion to withdraw the settlement, and on November 9, 2001 a motion was filed to dismiss FERC.  That same day, the plaintiffs filed a notice to voluntarily dismiss the claims.  No further activity appears on the docket.\n \nOn November 29, the defendant filed a response to the settlement of the case with the FEC.  A hearing was held on November 30, 2002.  Following the settlement negotiations, the Court held a conference call on November 8, 2002, and a hearing was scheduled for November 9.  While the conference was ongoing, the plaintiff filed a proposed settlement agreement with the defendants.  During the conference, the company filed a stipulation of dismissal of the claims against the defendants, which was granted by the FTC on December 18, 2002 and the terms were subsequently voluntarily dismissed.  Since the settlement was not resolved, the settlement is presumably closed.  For the time being, the litigation is ongoing and the case closed.\n\n\n\n\n   \n\nAs of April 2018, the status of the settlements is unknown.\n\n\n\n  On August 9 the parties filed a joint motion for a settlement agreement, which the court approved on December 20, 2018.  Regarding the settlement agreements, the agreement stated that it would not be binding on the defendants for the duration of the term of the terms.  Further, the consent decree was to remain in effect for the purpose of the final settlement agreement and the parties agreed to a payment of $1.5 million to the plaintiffs.  Finally, the three parties agreed that the settlement would be subject to a three-year extension of the consent agreement.\n\tThe case remains ongoing. \n\n\nWe have no additional information on the status on the litigation. documents filed by the parties are available to the group. <a href=\"http://www.gibbs-bruns.com/news/2017/08/09/2017-09-09_federal-lawsuit-filing-final-status-and-waiver-proposal-to-long- startup-time-unit\">here</a>.\n\nWe do not have a copy of the FCA settlement agreement; therefore, the dailies are presumably closed as of August 9. is the only available document in the dudge.  If the settlement does not resolve the dispute, the dispute is presumably resolved through a consent order.  But the dukedom of the parties is presumably still pending.\n<brief summary</b>\nThe docket indicates that the case remains open. is the last entry in the case for the settlement and the settlement discussions. has not yet been resolved. continues to monitor the settlement process. concludes the settlement talks. extends the settlement to August 9 and the dikedom of has been terminated. retains jurisdiction to enforce the terms, but we have no information on any further proceedings. maintains the case as of July 31, 2018 retains the right to monitor compliance. disputes over the settlement have continued to be resolved.\nThe settlement negotiations are ongoing.\n\nLaw agrees to payLawsuit."

        },
        {
            "id": 8,
            "title": ds_samples[8][SampleType.NAME],
            "source_text": ds_samples[8][SampleType.FULL_AS_STRING],
            "date": ds_samples[8][SampleType.KEY],
            "catchphrases": ds_samples[8][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/02/cauliflower-and-tofu-curry-document.html",
            "summary": "RE: risk 2001 follow up: risk-2001 follow up <a href=\"http://risk-2001.algo.com/risk-net/\">IM-ENRON</a></a>.\n\nWe have a follow-up meeting in Houston, TX on June 14, 2001.  We have no further information on the schedule.\n\n\nWe had a good time at risk 2001. The risk-net event was a great success. We had a great time discussing the various issues discussed in the event of a failure to meet. We were able to meet with a few people and arrange a follow up meeting in the Houston area.  \n\nOn June 14 and 16, 2001, we had a very good meeting. We discussed the issues discussed at risk- 2001 and the potential for a risk-to-risk 2001 follow-off meeting in TX. We also discussed the potential of a risk 2001 meeting in San Antonio, TX. \nWe will have to wait for the results of that meeting before we can meet in Houston.  As of June 30, 2001 we have no additional information on that matter.\nOn June 17, 2001 the risk-site meeting was cancelled.\nOn July 1, 2001 it was announced that the risk 2001 schedule had been cancelled. We have not yet met in Houston and have no plans to meet in person.On September 9, 2001 a followup meeting was arranged.The risk-sites are closed.We have no more information on this case.Views are closed, and we have not discussed.s is closed.\nViewings are closed and we do not have any further information.Scheduledscheduled a risk meeting in Texas on September 9.hundreds of attendees attended. ViewedShedding ensued.From:\n\nTo:\nFromFrom the Algo booth:The Risk 2001 followup:From a risk perspective,situated:On March 31, 2001 at risk, the risk site reported that the parties had reached a settlement. The settlement was not disclosed.This is a status update.As of March 31st, the parties have not reached a Settlement.Status update:Status reportStatus tracker:Activity:UpdatedActivityDate:DateOctober 1st, 2001Subject:SubjectBackground:BackgroundItem:ItemDocument:DocumentObject:ObjectPolicy:PolicyCollection:CollectionLocation:LocationDescription:\n\n\nItem Summary: Summary:\nItem:\nStatus::\t\n\n\t\nStatus:Status:\n\t:\n\n\n\n:\t\n\tStatus: Status:; Correspondence Forum; Correspondent forum<a1)Litigation(Updated:<hr>\n\nStatus updateDiscovery:\nUpdated:\nDate: June 14th, 2001 <a)Updated: <a>Updated: May 1stAsBrief:\nSummary:\nOn September 1st <a]\u2019s summary:\n>\nUpdated:\n<a>\nOn March 30th, 2017APossible:\nFrom risk 2001:\n <a summary:Summary: <hr>Updated<blockquote>\n\nUpdated Date:\nSubject:\nThe Risk 2000 follow upUpdated:\n\n inferredNo update:\n <blockSummary\t\n\t\tSummary\n\t\t\n\n\n\n\n\n\n\n\t\n\nSummary Summary\n \n\n\n\n\n\n\n\n\n\t\t Summary <a <\n> </a\n <i</a\t>\n\t\n\n\n\t> <a summary.</a>\n<block>\n<brief Summary: <i> Risk 2001 summary\nFrom\nDate\nStatus Update:\nGiven>\n\n\n>\nFrom: </blockquote------------------------------------------------\n\n>Summary>\nDate Summary:\n\n\nUpdated <a brief summary: <blockquote>\" Risk 2001 update: <brief summary:>\n <block </i>\n>Updated <i>, Summary </brief>\n</a>,Summary</a>)\n\n> <a>, <a current status update>\n </a> <i></a>\n\n <i/>\n\n <battery>\n \n\n </a href="

        },
        {
            "id": 9,
            "title": ds_samples[9][SampleType.NAME],
            "source_text": ds_samples[9][SampleType.FULL_AS_STRING],
            "date": ds_samples[9][SampleType.KEY],
            "catchphrases": ds_samples[9][SampleType.CATCHPHRASES],
            "url": "http://www.seriouseats.com/documents/2011/02/cauliflower-and-tofu-curry-document.html",
            "summary": "Re: Havamann Litigation PRIVILEGED AND CONFIDENTIAL ATTORNEY CLIENT  COMMUNICATION   \n\nOn January 30, 2000, the Havamanna Litigation Public Relations and Client Services (ECT) filed this lawsuit against the underwriters in the U.S. District Court for the District of Columbia. The case was assigned to the private counsel of the Havamusann Litigating Public Relations. The underwriters were the underwriter and the underwriting firm. The lawsuit was filed in the United States District Court of the District for the Eastern District of New York. The defendant underwriters had a claim against the defendant under the under-writers under the Havasann Litigants Act, the Act, and the Havamsann Lit. The plaintiffs alleged that the under underwriters failed to provide the settlement funds necessary to settle the case. The defendants also claimed that the defendant failed to transfer the settlement fund to the underwritten firm. \n \nOn February 14, the parties reached a settlement agreement. The parties agreed to a settlement conference.  \n\n \n\n\nThe parties agreed that the case would be closed in March. The parties are in the process of settling the case and have not yet reached a final settlement.  As of this writing, there is no further activity on the docket, and we have no further information. \n\n \nWe have no more information on the case, and no further docket activity.On February 16, the case was closed.On On February 17, the settlement was closed on the settlement The case is closed.  On February 14 and the parties are continuing to engage in settlement negotiations. On March 17, 2017, the court ordered the parties to file a joint motion for summary judgment.  The parties are still negotiating the terms of the settlement and the court has yet to rule on the motion.  We have no information on this case.\n   The case is ongoing.\n\n \n\n\n  On March 17th, the plaintiffs filed a motion for a preliminary injunction. and,., and  The motion is pending. |\u2022 \u2022- ----------------------------------------------------------------- SummarySummary Per; Ext ; )... --------------------...... ---- * // }................ I, v We! This Together /// </a>>> I am in London this week and will be in the UK during the week of February 14 if there is anything I can do to help is: Contact Agreement agrees statuss agreement agreed condition state subject settlement.  They agree but\u2019 asserted......\n. . See [\u2026]Brief( );<Litigation inferredWeC1Discovery'lIn this caseJGrief, </\u00a0\u00a0\u00a0\">AsFP7This case  This case </ </i>  and </o</l  </audio> </brief> > More <a.</ ### \" \\\t ##\n\n   The ( No \u00b6 To $ ---------------------------------------------------------------- Correspondent l All IllAbstract In \u00a7\u00a7 Child \u201c ll For []\\/\\/ largely UNCLASSIFIED Since Un Crim Significant\u00b7\u00b7 Log protracted)</ Environmental>>\\By StAll) FinancialA plaintiveHavamann Various ##### Transcript Settlement Claims -------- Given Regarding Attorneys-- --- Document Wireless Story Details Audio Law 2017 #To TitleAnd"

        },
    ]


class TestBert(unittest.TestCase):
    def setUp(self):
        # CREATE A ENCODER DOCODER MODEL
        self.model_cache = {}
        self.tokenizer_cache = {}

    def load_model_if_needed(self, model_cls: Type[BaseTransformer],
                             model_name: str, quantize_8bit: bool = False):
        if model_name not in self.model_cache:
            model_node: ModelNode = \
                ModelNode.load_from_id(model_cls,
                                       model_name,
                                       model_name, quantize_8bit=quantize_8bit)
            model_node.model.eval()
            self.model_cache[model_name] = model_node.model
            self.tokenizer_cache[model_name] = model_node.tokenizer
            settings.print_gpu_usage()
        return self.model_cache[model_name], self.tokenizer_cache[model_name]

    def test_tagging_embedding_controller_shape(self):
        model: LEDForConditionalGeneration
        model, tokenizer = self.load_model_if_needed(
            LEDForConditionalGeneration,
            'allenai/led-base-16384-multi_lexsum-source-long')

        tags = ['hello', 'ground']
        with torch.no_grad():
            embedding: FloatTensorT[
                'TaskSize,EmbedLen'] = \
                tagging_embedding_controller(model,
                                             ['hello world', 'hello world'],
                                             tags, tokenizer)
        assert 768 == model.embed_dim
        self.assertEqual(embedding.shape, (len(tags), 768))

    def test_tagging_embedding(self):
        with torch.cuda.amp.autocast(enabled=True):
            model: MistralForCausalLM
            model, tokenizer = self.load_model_if_needed(MistralForCausalLM,
                                                         'openchat/openchat_3.5',
                                                         quantize_8bit=True)
            docs = importing_sample_data(tokenizer)
            print(json.dumps(docs, indent=4))
            all_tags = set()
            for doc in docs:
                all_tags.update(doc['catchphrases'])
            print(all_tags)
            exit(3)

            tag_text = "Is this about cats?: "
            max_len = 4096
            with torch.no_grad():
                positive_examples = [tag_text + 'This is about cats',
                                     tag_text + 'This is about dogs']
                pos_embedding: FloatTensorT['EmbedLen'] = \
                    embedding_controller_from_texts(model, positive_examples,
                                                    tokenizer, max_src_len=max_len).mean(0,
                                                                              keepdim=False)
                settings.print_gpu_usage()

                negative_examples = [tag_text + 'This is about trains',
                                     tag_text + 'This is about cars']
                neg_embedding: FloatTensorT['EmbedLen'] = \
                    embedding_controller_from_texts(model, negative_examples,
                                                    tokenizer, max_src_len=max_len).mean(0,
                                                                              keepdim=False)

                support_embedding = FloatTensorT(
                    torch.stack([neg_embedding, pos_embedding], dim=0),
                    'NClasses,EmbedLen')

                del pos_embedding
                del neg_embedding
                settings.empty_cache()

                query_examples = [tag_text + 'Cats jump high',
                                  tag_text + 'Cars go fast']
                query_embedding: FloatTensorT[
                    'TaskSize,EmbedLen'] = \
                    embedding_controller_from_texts(model, query_examples, tokenizer,
                                                    max_src_len=max_len)

                probs = predict_with_centers(support_embedding, query_embedding)
                print(probs)
        self.assertEqual(probs.shape, (2, 2))


@torch.no_grad()
def test_summarization():
    logging.basicConfig(level=logging.WARNING)
    model_name = "/home/ceyer/Documents/Projects/LoA/backend/model_weights/tagging"
    # model_name = 'openchat/openchat_3.5'
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast(enabled=True):
        quantize = True
        quantization_cfg = None
        if quantize:
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes

                quantization_cfg = BnbQuantizationConfig(
                    load_in_8bit=quantize, )
            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None
                quantize = False

        model_node: ModelNode = \
            ModelNode.load_from_id(MistralForCausalLMHug, model_name,
                                   model_name,
                                   quantization_config=quantization_cfg)
        settings.print_gpu_usage()
        tokenizer = model_node.tokenizer
        model_node.model.eval()
    with ((torch.cuda.amp.autocast(enabled=True))):
        tag_qa_text = (
            """
                Message-ID: <23743848.1075863311776.JavaMail.evans@thyme>
                Date: Wed, 11 Jul 2001 08:29:22 -0700 (PDT)
                From: legalonline-compliance@enron.com
                To: williams@mailman.enron.com, bwillia5@enron.com
                Subject: Confidential Information and Securities Trading
                Mime-Version: 1.0
                Content-Type: text/plain; charset=us-ascii
                Content-Transfer-Encoding: 7bit
                X-From: Office of the Chairman - Enron Wholesale Services <legalonline-compliance@enron.com>@ENRON <IMCEANOTES-Office+20of+20the+20Chairman+20-+20Enron+20Wholesale+20Services+20+3Clegalonline-compliance+40enron+2Ecom+3E+40ENRON@ENRON.com>
                X-To: WILLIAMS@mailman.enron.com, WILLIAM <bwillia5@enron.com>
                X-cc: 
                X-bcc: 
                X-Folder: \Williams III, Bill (Non-Privileged)\Bill Williams III
                X-Origin: Williams-B
                X-FileName: Williams III, Bill (Non-Privileged).pst
    
                To:WILLIAMS, WILLIAM
                Email:bwillia5@enron.com - 503-464-3730
    
                Enron Wholesale Services - Office of the Chairman
    
                From:  Mark Frevert, Chairman & CEO
                Mark Haedicke, Managing Director & General Counsel
    
                Subject:  Confidential Information and Securities Trading
    
                To keep pace with the fluid and fast-changing demands of our equity trading activities, Enron Wholesale Services ("EWS") has recently revised its official Policies and Procedures Regarding Confidential Information and Securities Trading ("Policies and Procedures").  These revisions reflect two major developments: (1) our equity trading activities have been extended into the United Kingdom, and (2) in an effort to streamline the information flow process, the "Review Team" will play a more centralized role, so that the role of the "Resource Group" is no longer necessary.You are required to become familiar with, and to comply with, the Policies and Procedures.  The newly revised Policies and Procedures are available for your review on LegalOnline, the new intranet website maintained by the Enron Wholesale Services Legal Department.  Please click on the attached link to access LegalOnline:
                http://legalonline.corp.enron.com/chinesewall.asp 
    
                If you have already certified compliance with the Policies and Procedures during the 2001 calendar year, you need not re-certify at this time, although you are still required to to review and become familiar with the revised Policies and Procedures.  If you have not certified compliance with the Policies and Procedures during the 2001 calendar year, then you must do so within two weeks of your receipt of this message.  The LegalOnline site will allow you to quickly and conveniently certify your compliance on-line with your SAP Personal ID number.  If you have any questions concerning the Policies or Procedures, please call Bob Bruce at extension 5-7780 or Donna Lowry at extension 3-1939. """)
        text = [tag_qa_text, tag_qa_text + " shoot shoot"]
        summary = summarize_controller_from_texts(model_node.model, text, tokenizer, 4000)
        print(summary)
