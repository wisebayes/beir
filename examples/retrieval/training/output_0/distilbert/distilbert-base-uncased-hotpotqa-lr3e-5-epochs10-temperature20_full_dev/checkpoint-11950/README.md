---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:153000
- loss:MultipleNegativesRankingLoss
base_model: distilbert/distilbert-base-uncased
widget:
- source_sentence: The low cost airline known as Skymark Airlines is headquartered
    at what airport in Japan?
  sentences:
  - 'Tokyo International Airport (東京国際空港 , Tōkyō Kokusai Kūkō ) , commonly known as
    Haneda Airport (羽田空港 , Haneda Kūkō ) or Tokyo Haneda Airport (IATA: HND, ICAO:
    RJTT) , is one of the two primary airports that serve the Greater Tokyo Area,
    and is the primary base of Japan''s two major domestic airlines, Japan Airlines
    (Terminal 1) and All Nippon Airways (Terminal 2), as well as Air Do, Skymark Airlines,
    Solaseed Air, and StarFlyer. It is located in Ōta, Tokyo, 14 km south of Tokyo
    Station.'
  - Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was
    a German Army artillery officer whose career spanned World War I and World War
    II. He was a leader of Nazi Germany's V-2 rocket program and other projects at
    the Peenemünde Army Research Center.
  - Edgerton "Ed" Hartwell II (born May 27, 1978) is a former American football linebacker.
    He played college football at Western Illinois and was drafted by the Baltimore
    Ravens in the fourth round of the 2001 NFL Draft. Hartwell was also a member of
    the Atlanta Falcons, Cincinnati Bengals, Oakland Raiders and Las Vegas Locomotives.
- source_sentence: Who has been in more bands, Philip Labonte or Dave Williams?
  sentences:
  - Jedd Hughes (born in Quorn, Australia) is a singer, songwriter, session guitar
    player, and record producer.
  - Moves was a wargaming magazine originally published by SPI (Simulations Publications,
    Inc.), who also published manual wargames. Their flagship magazine "Strategy &
    Tactics" ("S&T"), was a military history magazine featuring a new wargame in each
    issue. While S&T was devoted to historical articles, "Moves" focused on the play
    of the games. Each issue carried articles dealing with strategies for different
    wargames, tactical tips, and many variants and scenarios for existing games. As
    time passed, reviews of new games also became an important feature. While the
    majority of the articles dealt with SPI games, the magazine was open to and published
    many articles on games by other companies.
  - Philip Steven Labonte (Born April 15, 1975) is an American musician from Massachusetts,
    best known as the lead singer of the American heavy metal band All That Remains.
    Labonte is the former lead vocalist for Shadows Fall, was the touring vocalist
    for Killswitch Engage in early 2010, and also filled in for Five Finger Death
    Punch vocalist Ivan L. Moody in late 2016.
- source_sentence: The University of Science and Arts of Oklahoma is funded by what
    system?
  sentences:
  - George Jasper Caster (August 4, 1907 – December 18, 1955), nicknamed "Ug", was
    a right-handed professional baseball pitcher for 21 years from 1929 to 1948 and
    again in 1953. He played 12 years in Major League Baseball with the Philadelphia
    Athletics (1934–1935, 1937–1940), St. Louis Browns (1941–1945), and Detroit Tigers
    (1945–1946).
  - The University of Science and Arts of Oklahoma, or USAO, is a public liberal arts
    college located in Chickasha, Oklahoma. It is the only public college in Oklahoma
    with a strictly liberal arts-focused curriculum and is a member of the Council
    of Public Liberal Arts Colleges. USAO is an undergraduate-only institution and
    grants Bachelor's Degrees in a variety of subject areas. The school was founded
    in 1908 as a school for women and from 1912 to 1965 was known as Oklahoma College
    for Women. It became coeducational in 1965 and today educates approximately 1,000
    students. In 2001, the entire Oklahoma College for Women campus was listed as
    a National Historic District.
  - '"Lisa the Greek" is the fourteenth episode of "The Simpsons"<nowiki>''</nowiki>
    third season. It originally aired on the Fox network in the United States on January
    23, 1992. In the episode, Homer begins to bond with his daughter, Lisa, after
    learning her unique and convenient ability to pick winning football teams, but,
    secretly, uses her ability to help him gamble. When Lisa finds out Homer''s secret,
    she refuses to speak to her father until he fully understands her. "Lisa the Greek"
    was written by Jay Kogen and Wallace Wolodarsky, and directed by Rich Moore.'
- source_sentence: The stepchild of Shirley Jones turned down a song later recorded
    by what band?
  sentences:
  - David Bruce Cassidy (born April 12, 1950) is a retired American actor, singer,
    songwriter, and guitarist. He is known for his role as Keith Partridge, the son
    of Shirley Partridge (played by his stepmother Shirley Jones), in the 1970s musical-sitcom
    "The Partridge Family", which led to his becoming one of popular culture's teen
    idols and pop singers of the 1970s. He later had a career in both acting and music.
  - Hal Hartley (born November 3, 1959) is an American film director, screenwriter,
    producer and composer who became a key figure in the American independent film
    movement of the 1980s and '90s. He is best known for his films "Trust", "Amateur"
    and "Henry Fool", which are notable for deadpan humour and offbeat characters
    quoting philosophical dialogue.
  - Str8 Outta Northcote is the second full-length album from Australian grindcore
    band Blood Duster, its title being a parody of the N.W.A album "Straight Outta
    Compton". Northcote is the suburb in Melbourne where most of the band lived at
    the time.
- source_sentence: Which district of Lancashire, England contains the village that
    lies between Langho and a village that is on the banks of the River Calder?
  sentences:
  - James Lawrence (16 February 1879 – 21 November 1934) was a Scottish football player
    and manager. A goalkeeper, he played for Newcastle United between 1904 and 1922.
  - Billington is a village in the Ribble Valley district of Lancashire, England.
    It lies between the villages of Whalley and Langho. It forms part of the Billington
    and Langho civil parish and contains the schools St Augustine's RC High School,
    St Leonard's Primary and St Mary's Primary.
  - The Orange Tundra is a cocktail of Vodka, Kahlúa, Creme Soda, and Orange juice.
    It is traditionally poured over 2 to 3 cubes of ice in an old-fashioned 8-12 oz.
    glass.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on distilbert/distilbert-base-uncased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) <!-- at revision 12040accade4e8a0f71eabdb258fecc2e7e948be -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Energy Distance
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Which district of Lancashire, England contains the village that lies between Langho and a village that is on the banks of the River Calder?',
    "Billington is a village in the Ribble Valley district of Lancashire, England. It lies between the villages of Whalley and Langho. It forms part of the Billington and Langho civil parish and contains the schools St Augustine's RC High School, St Leonard's Primary and St Mary's Primary.",
    'James Lawrence (16 February 1879 – 21 November 1934) was a Scottish football player and manager. A goalkeeper, he played for Newcastle United between 1904 and 1922.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 153,000 training samples
* Columns: <code>query</code> and <code>text</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                              | text                                                                                |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 7 tokens</li><li>mean: 24.16 tokens</li><li>max: 117 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 95.71 tokens</li><li>max: 302 tokens</li></ul> |
* Samples:
  | query                                                                                                                                                          | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>The surf reef break which Joel Parkinson surfed on in the ASP World Championship Tour Surfing Title on 14 December 2012 is located in what state?</code> | <code>Joel Parkinson (born April 10, 1981) is an Australian surfer who competes on the ASP (Association Of Surfing Professionals) World Tour. After twelve years competing at the elite level on the ASP World Championship Tour, a stretch that saw him win eleven elite ASP World Title Events, plus nine additional ASP tour events, and achieve runner-up second place to the ASP World Title four times, Parkinson won the ASP World Championship Tour Surfing Title on 14 December 2012 in Hawaii at the Banzai Pipeline during the ASP World Tours' final event for 2012–the Billabong Pipeline Masters. Parkinson hung on in a back and forth battle with eleven-time ASP World Title holder, Kelly Slater, to get his first World Title, as well as go on to win the Pipeline Masters, only after Slater lost his semi-final heat to Josh Kerr, of Queensland, Australia. Parkinson beat Kerr in the finals of the event, which was his seventh top-five placing for the year, and his first event title win for 2012.</code> |
  | <code>What type of publications are both Woman's World and Pictorial Review?</code>                                                                            | <code>Woman's World is an American supermarket weekly magazine with a circulation of 1.6 million readers. Printed on paper generally associated with tabloid publications and priced accordingly, it concentrates on short articles about subjects such as weight loss, relationship advice and cooking, along with feature stories about women in the STEM fields and academia. It has held the title of the most popular newsstand women's magazine, with sales of 77 million copies in 2004. It competes with more general-market traditional magazines such as "Woman's Day" and "Family Circle".</code>                                                                                                                                                                                                                                                                                                                                                                                                                           |
  | <code>Anna Cruz plays basketball for the WNBA team located in what city?</code>                                                                                | <code>Anna Cruz Lebrato (born October 27, 1986) is a Spanish professional basketball player with the Minnesota Lynx of the WNBA. She plays for Nadezhda Orenburg, Minnesota Lynx and Spain women's national basketball team. She has represented national team since 2009, earning bronze medals at EuroBasket Women 2009, and World Championship 2010 and silver medal at World Championship 2014.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "<lambda>"
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 17,000 evaluation samples
* Columns: <code>query</code> and <code>text</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                              | text                                                                                |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 8 tokens</li><li>mean: 24.62 tokens</li><li>max: 129 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 95.83 tokens</li><li>max: 387 tokens</li></ul> |
* Samples:
  | query                                                                                                                                                                                            | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Were A Funny Thing Happened on the Way to the Moon and Air Force, Incorporated directed by the same person?</code>                                                                         | <code>A Funny Thing Happened on the Way to the Moon is a 2001 film written, produced and directed by Nashville-based filmmaker Bart Sibrel. Sibrel is a critic of the Apollo program and proponent of the conspiracy theory that the six Apollo Moon landing missions between 1969 and 1972 were elaborate hoaxes perpetrated by the United States government, including NASA.</code>                                                                                                                                                                                                                                                                    |
  | <code>Which gold mine, Camlaren Mine or Burwash Mine is located farther north of Yellowknife?</code>                                                                                             | <code>The Burwash Mine was a small gold property discovered in the fall of 1934 by Johnny Baker and Hugh Muir at Yellowknife Bay, Northwest Territories. The town of Yellowknife did not exist yet at that point, but the discovery of gold at Burwash was the catalyst that brought more gold prospectors into the region in 1935 and 1936. A short shaft was sunk in 1935-1936 at Burwash, and in the summer of 1935 a 16-ton bulk sample of ore was shipped to Trail, British Columbia for processing, yielding 200 troy ounces (6 kg) of gold. The mine did not become a substantial producer and it is believed the gold vein was mined out.</code> |
  | <code>The MKT Nature and Fitness Trail runs nine miles in the right of way of what former Class 1 railroad company that was established in 1865, and had its last headquarters in Dallas?</code> | <code>The Missouri–Kansas–Texas Railroad (reporting mark MKT) is a former Class I railroad company in the United States, with its last headquarters in Dallas. Established in 1865 under the name Union Pacific Railway, Southern Branch, it came to serve an extensive rail network in Texas, Oklahoma, Kansas, and Missouri. In 1988, it merged with the Missouri Pacific Railroad and is now part of Union Pacific Railroad.</code>                                                                                                                                                                                                                   |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "<lambda>"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `gradient_accumulation_steps`: 4
- `learning_rate`: 3e-05
- `weight_decay`: 0.01
- `num_train_epochs`: 10
- `warmup_steps`: 956
- `seed`: 24
- `fp16`: True
- `ddp_find_unused_parameters`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 4
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 3e-05
- `weight_decay`: 0.01
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 956
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 24
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: False
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0209 | 25    | 7287.04       |
| 0.0418 | 50    | 6615.5337     |
| 0.0627 | 75    | 5165.5994     |
| 0.0837 | 100   | 3156.7488     |
| 0.1046 | 125   | 1752.7506     |
| 0.1255 | 150   | 922.3852      |
| 0.1464 | 175   | 468.4091      |
| 0.1673 | 200   | 252.4444      |
| 0.1882 | 225   | 144.2602      |
| 0.2092 | 250   | 82.9497       |
| 0.2301 | 275   | 55.2185       |
| 0.2510 | 300   | 42.692        |
| 0.2719 | 325   | 34.3373       |
| 0.2928 | 350   | 30.8981       |
| 0.3137 | 375   | 26.751        |
| 0.3347 | 400   | 23.1121       |
| 0.3556 | 425   | 20.071        |
| 0.3765 | 450   | 18.7708       |
| 0.3974 | 475   | 17.1152       |
| 0.4183 | 500   | 15.2941       |
| 0.4392 | 525   | 14.1917       |
| 0.4602 | 550   | 13.2238       |
| 0.4811 | 575   | 12.623        |
| 0.5020 | 600   | 12.3279       |
| 0.5229 | 625   | 11.87         |
| 0.5438 | 650   | 11.7334       |
| 0.5647 | 675   | 11.3767       |
| 0.5857 | 700   | 10.7534       |
| 0.6066 | 725   | 10.8914       |
| 0.6275 | 750   | 10.4907       |
| 0.6484 | 775   | 10.3467       |
| 0.6693 | 800   | 9.7345        |
| 0.6902 | 825   | 9.6149        |
| 0.7111 | 850   | 9.3884        |
| 0.7321 | 875   | 12.5412       |
| 0.7530 | 900   | 8.4376        |
| 0.7739 | 925   | 8.1836        |
| 0.7948 | 950   | 7.7423        |
| 0.8157 | 975   | 7.6154        |
| 0.8366 | 1000  | 7.2809        |
| 0.8576 | 1025  | 7.0679        |
| 0.8785 | 1050  | 6.5942        |
| 0.8994 | 1075  | 6.6448        |
| 0.9203 | 1100  | 6.4179        |
| 0.9412 | 1125  | 6.353         |
| 0.9621 | 1150  | 6.1897        |
| 0.9831 | 1175  | 5.9103        |
| 1.0033 | 1200  | 5.7641        |
| 1.0243 | 1225  | 5.7074        |
| 1.0452 | 1250  | 5.6942        |
| 1.0661 | 1275  | 7.0173        |
| 1.0870 | 1300  | 5.5911        |
| 1.1079 | 1325  | 5.2273        |
| 1.1288 | 1350  | 5.3551        |
| 1.1498 | 1375  | 4.8977        |
| 1.1707 | 1400  | 4.8689        |
| 1.1916 | 1425  | 5.015         |
| 1.2125 | 1450  | 4.7961        |
| 1.2334 | 1475  | 4.6335        |
| 1.2543 | 1500  | 4.667         |
| 1.2753 | 1525  | 4.5171        |
| 1.2962 | 1550  | 4.4934        |
| 1.3171 | 1575  | 4.1949        |
| 1.3380 | 1600  | 3.9994        |
| 1.3589 | 1625  | 4.2289        |
| 1.3798 | 1650  | 4.0993        |
| 1.4008 | 1675  | 4.2167        |
| 1.4217 | 1700  | 4.184         |
| 1.4426 | 1725  | 3.9625        |
| 1.4635 | 1750  | 3.8333        |
| 1.4844 | 1775  | 3.6684        |
| 1.5053 | 1800  | 3.9582        |
| 1.5262 | 1825  | 4.2164        |
| 1.5472 | 1850  | 3.8607        |
| 1.5681 | 1875  | 4.2899        |
| 1.5890 | 1900  | 3.7762        |
| 1.6099 | 1925  | 3.6036        |
| 1.6308 | 1950  | 3.7383        |
| 1.6517 | 1975  | 3.7085        |
| 1.6727 | 2000  | 3.6817        |
| 1.6936 | 2025  | 3.3364        |
| 1.7145 | 2050  | 3.6588        |
| 1.7354 | 2075  | 3.6554        |
| 1.7563 | 2100  | 3.4816        |
| 1.7772 | 2125  | 3.3675        |
| 1.7982 | 2150  | 3.4066        |
| 1.8191 | 2175  | 3.3328        |
| 1.8400 | 2200  | 3.5218        |
| 1.8609 | 2225  | 3.3473        |
| 1.8818 | 2250  | 3.5968        |
| 1.9027 | 2275  | 3.2225        |
| 1.9237 | 2300  | 3.3081        |
| 1.9446 | 2325  | 3.4868        |
| 1.9655 | 2350  | 3.2818        |
| 1.9864 | 2375  | 3.2902        |
| 2.0067 | 2400  | 5.2864        |
| 2.0276 | 2425  | 2.927         |
| 2.0485 | 2450  | 2.9363        |
| 2.0694 | 2475  | 3.0663        |
| 2.0904 | 2500  | 3.269         |
| 2.1113 | 2525  | 3.024         |
| 2.1322 | 2550  | 2.9208        |
| 2.1531 | 2575  | 3.1275        |
| 2.1740 | 2600  | 2.9161        |
| 2.1949 | 2625  | 2.8242        |
| 2.2159 | 2650  | 2.8874        |
| 2.2368 | 2675  | 3.0466        |
| 2.2577 | 2700  | 2.6925        |
| 2.2786 | 2725  | 2.8583        |
| 2.2995 | 2750  | 2.7241        |
| 2.3204 | 2775  | 2.6911        |
| 2.3414 | 2800  | 2.7545        |
| 2.3623 | 2825  | 2.7682        |
| 2.3832 | 2850  | 2.7545        |
| 2.4041 | 2875  | 2.5287        |
| 2.4250 | 2900  | 2.6729        |
| 2.4459 | 2925  | 2.7219        |
| 2.4668 | 2950  | 2.6623        |
| 2.4878 | 2975  | 2.6815        |
| 2.5087 | 3000  | 2.7268        |
| 2.5296 | 3025  | 2.7212        |
| 2.5505 | 3050  | 2.7546        |
| 2.5714 | 3075  | 2.5527        |
| 2.5923 | 3100  | 2.6804        |
| 2.6133 | 3125  | 2.7775        |
| 2.6342 | 3150  | 2.6498        |
| 2.6551 | 3175  | 2.5718        |
| 2.6760 | 3200  | 2.6391        |
| 2.6969 | 3225  | 2.7538        |
| 2.7178 | 3250  | 2.6966        |
| 2.7388 | 3275  | 2.4594        |
| 2.7597 | 3300  | 2.6976        |
| 2.7806 | 3325  | 2.486         |
| 2.8015 | 3350  | 2.6426        |
| 2.8224 | 3375  | 2.5044        |
| 2.8433 | 3400  | 2.6027        |
| 2.8643 | 3425  | 2.4501        |
| 2.8852 | 3450  | 2.47          |
| 2.9061 | 3475  | 2.5113        |
| 2.9270 | 3500  | 3.4089        |
| 2.9479 | 3525  | 2.5679        |
| 2.9688 | 3550  | 2.585         |
| 2.9898 | 3575  | 2.3694        |
| 3.0100 | 3600  | 2.2647        |
| 3.0310 | 3625  | 2.2014        |
| 3.0519 | 3650  | 2.3167        |
| 3.0728 | 3675  | 2.4738        |
| 3.0937 | 3700  | 2.4334        |
| 3.1146 | 3725  | 2.3451        |
| 3.1355 | 3750  | 2.2832        |
| 3.1565 | 3775  | 2.1955        |
| 3.1774 | 3800  | 2.1188        |
| 3.1983 | 3825  | 2.3287        |
| 3.2192 | 3850  | 2.4039        |
| 3.2401 | 3875  | 2.3324        |
| 3.2610 | 3900  | 5.3639        |
| 3.2819 | 3925  | 2.294         |
| 3.3029 | 3950  | 2.2803        |
| 3.3238 | 3975  | 2.3673        |
| 3.3447 | 4000  | 2.367         |
| 3.3656 | 4025  | 2.2887        |
| 3.3865 | 4050  | 2.1779        |
| 3.4074 | 4075  | 2.1668        |
| 3.4284 | 4100  | 2.2043        |
| 3.4493 | 4125  | 2.2458        |
| 3.4702 | 4150  | 2.0272        |
| 3.4911 | 4175  | 2.2069        |
| 3.5120 | 4200  | 2.1971        |
| 3.5329 | 4225  | 2.1678        |
| 3.5539 | 4250  | 2.2503        |
| 3.5748 | 4275  | 2.1468        |
| 3.5957 | 4300  | 2.1491        |
| 3.6166 | 4325  | 2.4192        |
| 3.6375 | 4350  | 2.0993        |
| 3.6584 | 4375  | 2.2519        |
| 3.6794 | 4400  | 2.2806        |
| 3.7003 | 4425  | 2.0965        |
| 3.7212 | 4450  | 2.2136        |
| 3.7421 | 4475  | 2.1663        |
| 3.7630 | 4500  | 2.0991        |
| 3.7839 | 4525  | 2.2654        |
| 3.8049 | 4550  | 2.3528        |
| 3.8258 | 4575  | 2.0346        |
| 3.8467 | 4600  | 2.0205        |
| 3.8676 | 4625  | 2.0359        |
| 3.8885 | 4650  | 2.0412        |
| 3.9094 | 4675  | 2.0971        |
| 3.9303 | 4700  | 2.0284        |
| 3.9513 | 4725  | 2.1545        |
| 3.9722 | 4750  | 2.247         |
| 3.9931 | 4775  | 2.0696        |
| 4.0134 | 4800  | 2.0553        |
| 4.0343 | 4825  | 1.847         |
| 4.0552 | 4850  | 2.1437        |
| 4.0761 | 4875  | 1.9105        |
| 4.0971 | 4900  | 1.9564        |
| 4.1180 | 4925  | 1.8703        |
| 4.1389 | 4950  | 1.7799        |
| 4.1598 | 4975  | 1.8773        |
| 4.1807 | 5000  | 2.0439        |
| 4.2016 | 5025  | 1.7843        |
| 4.2225 | 5050  | 1.8648        |
| 4.2435 | 5075  | 1.8777        |
| 4.2644 | 5100  | 1.9631        |
| 4.2853 | 5125  | 1.9279        |
| 4.3062 | 5150  | 1.9024        |
| 4.3271 | 5175  | 1.9858        |
| 4.3480 | 5200  | 1.9682        |
| 4.3690 | 5225  | 1.932         |
| 4.3899 | 5250  | 1.8638        |
| 4.4108 | 5275  | 1.9223        |
| 4.4317 | 5300  | 1.8602        |
| 4.4526 | 5325  | 2.0198        |
| 4.4735 | 5350  | 1.866         |
| 4.4945 | 5375  | 1.944         |
| 4.5154 | 5400  | 2.0164        |
| 4.5363 | 5425  | 1.8726        |
| 4.5572 | 5450  | 1.9949        |
| 4.5781 | 5475  | 1.8207        |
| 4.5990 | 5500  | 1.8883        |
| 4.6200 | 5525  | 1.9136        |
| 4.6409 | 5550  | 1.7779        |
| 4.6618 | 5575  | 1.833         |
| 4.6827 | 5600  | 1.8151        |
| 4.7036 | 5625  | 1.646         |
| 4.7245 | 5650  | 1.8242        |
| 4.7455 | 5675  | 1.7112        |
| 4.7664 | 5700  | 1.894         |
| 4.7873 | 5725  | 1.9538        |
| 4.8082 | 5750  | 1.8436        |
| 4.8291 | 5775  | 1.7558        |
| 4.8500 | 5800  | 1.7309        |
| 4.8709 | 5825  | 1.8367        |
| 4.8919 | 5850  | 1.7867        |
| 4.9128 | 5875  | 1.7639        |
| 4.9337 | 5900  | 1.7479        |
| 4.9546 | 5925  | 1.796         |
| 4.9755 | 5950  | 1.7172        |
| 4.9964 | 5975  | 1.8327        |
| 5.0167 | 6000  | 1.5956        |
| 5.0376 | 6025  | 1.6949        |
| 5.0586 | 6050  | 1.6426        |
| 5.0795 | 6075  | 1.7135        |
| 5.1004 | 6100  | 1.778         |
| 5.1213 | 6125  | 1.7899        |
| 5.1422 | 6150  | 1.5925        |
| 5.1631 | 6175  | 1.6831        |
| 5.1841 | 6200  | 1.7188        |
| 5.2050 | 6225  | 1.5955        |
| 5.2259 | 6250  | 1.7045        |
| 5.2468 | 6275  | 1.6516        |
| 5.2677 | 6300  | 1.6744        |
| 5.2886 | 6325  | 1.6891        |
| 5.3096 | 6350  | 1.6751        |
| 5.3305 | 6375  | 1.5709        |
| 5.3514 | 6400  | 1.6383        |
| 5.3723 | 6425  | 1.6177        |
| 5.3932 | 6450  | 1.6315        |
| 5.4141 | 6475  | 1.6348        |
| 5.4351 | 6500  | 1.6277        |
| 5.4560 | 6525  | 1.5968        |
| 5.4769 | 6550  | 1.7194        |
| 5.4978 | 6575  | 1.6061        |
| 5.5187 | 6600  | 1.635         |
| 5.5396 | 6625  | 1.5818        |
| 5.5606 | 6650  | 1.7574        |
| 5.5815 | 6675  | 1.6267        |
| 5.6024 | 6700  | 1.5892        |
| 5.6233 | 6725  | 1.5404        |
| 5.6442 | 6750  | 1.5757        |
| 5.6651 | 6775  | 1.6779        |
| 5.6860 | 6800  | 1.6268        |
| 5.7070 | 6825  | 1.4921        |
| 5.7279 | 6850  | 1.4954        |
| 5.7488 | 6875  | 1.6741        |
| 5.7697 | 6900  | 1.5385        |
| 5.7906 | 6925  | 1.6107        |
| 5.8115 | 6950  | 1.6327        |
| 5.8325 | 6975  | 1.7204        |
| 5.8534 | 7000  | 1.5499        |
| 5.8743 | 7025  | 1.5566        |
| 5.8952 | 7050  | 1.6524        |
| 5.9161 | 7075  | 1.5043        |
| 5.9370 | 7100  | 1.6936        |
| 5.9580 | 7125  | 1.6063        |
| 5.9789 | 7150  | 1.5914        |
| 5.9998 | 7175  | 1.5116        |
| 6.0201 | 7200  | 1.3606        |
| 6.0410 | 7225  | 1.3704        |
| 6.0619 | 7250  | 1.467         |
| 6.0828 | 7275  | 1.4476        |
| 6.1037 | 7300  | 1.4177        |
| 6.1247 | 7325  | 1.4575        |
| 6.1456 | 7350  | 1.4257        |
| 6.1665 | 7375  | 1.4598        |
| 6.1874 | 7400  | 1.4693        |
| 6.2083 | 7425  | 1.4532        |
| 6.2292 | 7450  | 1.5192        |
| 6.2502 | 7475  | 1.4303        |
| 6.2711 | 7500  | 1.5257        |
| 6.2920 | 7525  | 1.4049        |
| 6.3129 | 7550  | 1.3723        |
| 6.3338 | 7575  | 1.5183        |
| 6.3547 | 7600  | 1.3412        |
| 6.3757 | 7625  | 1.3675        |
| 6.3966 | 7650  | 1.2623        |
| 6.4175 | 7675  | 1.4986        |
| 6.4384 | 7700  | 1.4454        |
| 6.4593 | 7725  | 1.4244        |
| 6.4802 | 7750  | 1.3917        |
| 6.5012 | 7775  | 1.4485        |
| 6.5221 | 7800  | 1.476         |
| 6.5430 | 7825  | 1.493         |
| 6.5639 | 7850  | 1.4509        |
| 6.5848 | 7875  | 1.4726        |
| 6.6057 | 7900  | 1.446         |
| 6.6266 | 7925  | 1.4273        |
| 6.6476 | 7950  | 1.3591        |
| 6.6685 | 7975  | 1.4276        |
| 6.6894 | 8000  | 1.4159        |
| 6.7103 | 8025  | 1.5546        |
| 6.7312 | 8050  | 1.4129        |
| 6.7521 | 8075  | 1.3446        |
| 6.7731 | 8100  | 1.3257        |
| 6.7940 | 8125  | 1.5271        |
| 6.8149 | 8150  | 1.4054        |
| 6.8358 | 8175  | 1.3635        |
| 6.8567 | 8200  | 1.4694        |
| 6.8776 | 8225  | 1.4291        |
| 6.8986 | 8250  | 1.4389        |
| 6.9195 | 8275  | 1.4349        |
| 6.9404 | 8300  | 1.388         |
| 6.9613 | 8325  | 1.3483        |
| 6.9822 | 8350  | 1.434         |
| 7.0025 | 8375  | 1.3971        |
| 7.0234 | 8400  | 1.3085        |
| 7.0443 | 8425  | 1.316         |
| 7.0653 | 8450  | 1.3715        |
| 7.0862 | 8475  | 1.2874        |
| 7.1071 | 8500  | 1.2108        |
| 7.1280 | 8525  | 1.3213        |
| 7.1489 | 8550  | 1.3196        |
| 7.1698 | 8575  | 1.3462        |
| 7.1908 | 8600  | 1.1862        |
| 7.2117 | 8625  | 1.4343        |
| 7.2326 | 8650  | 1.2938        |
| 7.2535 | 8675  | 1.2226        |
| 7.2744 | 8700  | 1.3444        |
| 7.2953 | 8725  | 1.2028        |
| 7.3163 | 8750  | 1.2297        |
| 7.3372 | 8775  | 1.3571        |
| 7.3581 | 8800  | 1.2545        |
| 7.3790 | 8825  | 1.2806        |
| 7.3999 | 8850  | 1.3103        |
| 7.4208 | 8875  | 1.4025        |
| 7.4417 | 8900  | 1.2706        |
| 7.4627 | 8925  | 1.2854        |
| 7.4836 | 8950  | 1.3214        |
| 7.5045 | 8975  | 1.3055        |
| 7.5254 | 9000  | 1.3401        |
| 7.5463 | 9025  | 1.4002        |
| 7.5672 | 9050  | 1.2843        |
| 7.5882 | 9075  | 1.1982        |
| 7.6091 | 9100  | 1.2221        |
| 7.6300 | 9125  | 1.1549        |
| 7.6509 | 9150  | 1.267         |
| 7.6718 | 9175  | 1.2565        |
| 7.6927 | 9200  | 1.2828        |
| 7.7137 | 9225  | 1.3383        |
| 7.7346 | 9250  | 1.2643        |
| 7.7555 | 9275  | 1.231         |
| 7.7764 | 9300  | 1.2377        |
| 7.7973 | 9325  | 1.279         |
| 7.8182 | 9350  | 1.2965        |
| 7.8392 | 9375  | 1.3587        |
| 7.8601 | 9400  | 1.3332        |
| 7.8810 | 9425  | 1.2737        |
| 7.9019 | 9450  | 1.1891        |
| 7.9228 | 9475  | 1.206         |
| 7.9437 | 9500  | 1.2518        |
| 7.9647 | 9525  | 1.3168        |
| 7.9856 | 9550  | 1.1845        |
| 8.0059 | 9575  | 1.2343        |
| 8.0268 | 9600  | 1.1953        |
| 8.0477 | 9625  | 1.2073        |
| 8.0686 | 9650  | 1.1798        |
| 8.0895 | 9675  | 1.1719        |
| 8.1104 | 9700  | 1.1753        |
| 8.1314 | 9725  | 1.2755        |
| 8.1523 | 9750  | 1.1583        |
| 8.1732 | 9775  | 1.1742        |
| 8.1941 | 9800  | 1.1908        |
| 8.2150 | 9825  | 1.1943        |
| 8.2359 | 9850  | 1.1577        |
| 8.2569 | 9875  | 1.2108        |
| 8.2778 | 9900  | 1.2176        |
| 8.2987 | 9925  | 1.19          |
| 8.3196 | 9950  | 1.1343        |
| 8.3405 | 9975  | 1.2175        |
| 8.3614 | 10000 | 1.0908        |
| 8.3823 | 10025 | 1.202         |
| 8.4033 | 10050 | 1.3879        |
| 8.4242 | 10075 | 1.147         |
| 8.4451 | 10100 | 1.1367        |
| 8.4660 | 10125 | 1.1745        |
| 8.4869 | 10150 | 1.1737        |
| 8.5078 | 10175 | 1.1559        |
| 8.5288 | 10200 | 1.1333        |
| 8.5497 | 10225 | 1.1929        |
| 8.5706 | 10250 | 1.1663        |
| 8.5915 | 10275 | 1.1664        |
| 8.6124 | 10300 | 1.1578        |
| 8.6333 | 10325 | 1.2935        |
| 8.6543 | 10350 | 1.0937        |
| 8.6752 | 10375 | 1.1697        |
| 8.6961 | 10400 | 1.2246        |
| 8.7170 | 10425 | 1.0991        |
| 8.7379 | 10450 | 1.1856        |
| 8.7588 | 10475 | 1.0603        |
| 8.7798 | 10500 | 1.1805        |
| 8.8007 | 10525 | 1.228         |
| 8.8216 | 10550 | 1.1443        |
| 8.8425 | 10575 | 1.2603        |
| 8.8634 | 10600 | 1.2607        |
| 8.8843 | 10625 | 1.1796        |
| 8.9052 | 10650 | 1.0707        |
| 8.9262 | 10675 | 1.022         |
| 8.9471 | 10700 | 1.219         |
| 8.9680 | 10725 | 1.1911        |
| 8.9889 | 10750 | 1.1141        |
| 9.0092 | 10775 | 1.1967        |
| 9.0301 | 10800 | 1.0752        |
| 9.0510 | 10825 | 1.1054        |
| 9.0720 | 10850 | 1.1113        |
| 9.0929 | 10875 | 1.0617        |
| 9.1138 | 10900 | 1.0703        |
| 9.1347 | 10925 | 1.2254        |
| 9.1556 | 10950 | 1.0755        |
| 9.1765 | 10975 | 1.1226        |
| 9.1974 | 11000 | 1.0661        |
| 9.2184 | 11025 | 1.1592        |
| 9.2393 | 11050 | 1.2054        |
| 9.2602 | 11075 | 1.0501        |
| 9.2811 | 11100 | 1.0369        |
| 9.3020 | 11125 | 1.1677        |
| 9.3229 | 11150 | 1.068         |
| 9.3439 | 11175 | 0.9578        |
| 9.3648 | 11200 | 1.1794        |
| 9.3857 | 11225 | 1.0591        |
| 9.4066 | 11250 | 1.1041        |
| 9.4275 | 11275 | 1.1211        |
| 9.4484 | 11300 | 1.0642        |
| 9.4694 | 11325 | 1.1179        |
| 9.4903 | 11350 | 1.1357        |
| 9.5112 | 11375 | 1.0903        |
| 9.5321 | 11400 | 1.1634        |
| 9.5530 | 11425 | 1.0832        |
| 9.5739 | 11450 | 1.1346        |
| 9.5949 | 11475 | 1.1134        |
| 9.6158 | 11500 | 1.0217        |
| 9.6367 | 11525 | 1.0595        |
| 9.6576 | 11550 | 1.0886        |
| 9.6785 | 11575 | 1.0514        |
| 9.6994 | 11600 | 1.1276        |
| 9.7204 | 11625 | 1.1272        |
| 9.7413 | 11650 | 1.1155        |
| 9.7622 | 11675 | 1.0859        |
| 9.7831 | 11700 | 1.1246        |
| 9.8040 | 11725 | 1.133         |
| 9.8249 | 11750 | 1.0952        |
| 9.8458 | 11775 | 1.1181        |
| 9.8668 | 11800 | 1.0787        |
| 9.8877 | 11825 | 1.0798        |
| 9.9086 | 11850 | 1.0218        |
| 9.9295 | 11875 | 1.1141        |
| 9.9504 | 11900 | 1.1062        |
| 9.9713 | 11925 | 1.0237        |
| 9.9923 | 11950 | 1.1981        |

</details>

### Framework Versions
- Python: 3.9.21
- Sentence Transformers: 3.5.0.dev0
- Transformers: 4.51.3
- PyTorch: 2.1.2+cu121
- Accelerate: 1.6.0
- Datasets: 2.21.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->