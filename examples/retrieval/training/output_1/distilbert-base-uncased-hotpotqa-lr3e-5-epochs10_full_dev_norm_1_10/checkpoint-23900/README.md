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
  (2): Normalize()
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

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `gradient_accumulation_steps`: 4
- `learning_rate`: 3e-05
- `weight_decay`: 0.01
- `num_train_epochs`: 20
- `warmup_steps`: 956
- `seed`: 24
- `fp16`: True
- `ddp_find_unused_parameters`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `num_train_epochs`: 20
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

| Epoch   | Step  | Training Loss | Validation Loss |
|:-------:|:-----:|:-------------:|:---------------:|
| 0.0209  | 25    | 614.4612      | -               |
| 0.0418  | 50    | 544.868       | -               |
| 0.0627  | 75    | 410.5588      | -               |
| 0.0837  | 100   | 246.8498      | -               |
| 0.1046  | 125   | 128.7299      | -               |
| 0.1255  | 150   | 59.0294       | -               |
| 0.1464  | 175   | 24.9046       | -               |
| 0.1673  | 200   | 14.0487       | -               |
| 0.1882  | 225   | 11.9214       | -               |
| 0.2092  | 250   | 11.4853       | -               |
| 0.2301  | 275   | 11.337        | -               |
| 0.2510  | 300   | 11.2873       | -               |
| 0.2719  | 325   | 11.2497       | -               |
| 0.2928  | 350   | 11.2389       | -               |
| 0.3137  | 375   | 11.2119       | -               |
| 0.3347  | 400   | 11.1853       | -               |
| 0.3556  | 425   | 11.2057       | -               |
| 0.3765  | 450   | 11.1673       | -               |
| 0.3974  | 475   | 11.1487       | -               |
| 0.4183  | 500   | 11.1376       | -               |
| 0.4392  | 525   | 11.1123       | -               |
| 0.4602  | 550   | 11.1075       | -               |
| 0.4811  | 575   | 11.0876       | -               |
| 0.5020  | 600   | 11.0598       | -               |
| 0.5229  | 625   | 11.0428       | -               |
| 0.5438  | 650   | 11.0247       | -               |
| 0.5647  | 675   | 10.8647       | -               |
| 0.5857  | 700   | 10.3901       | -               |
| 0.6066  | 725   | 10.077        | -               |
| 0.6275  | 750   | 9.5194        | -               |
| 0.6484  | 775   | 9.4812        | -               |
| 0.6693  | 800   | 9.1373        | -               |
| 0.6902  | 825   | 8.606         | -               |
| 0.7111  | 850   | 8.5544        | -               |
| 0.7321  | 875   | 8.2767        | -               |
| 0.7530  | 900   | 8.3528        | -               |
| 0.7739  | 925   | 8.3404        | -               |
| 0.7948  | 950   | 8.1798        | -               |
| 0.8157  | 975   | 7.9366        | -               |
| 0.8366  | 1000  | 7.8792        | -               |
| 0.8576  | 1025  | 7.6574        | -               |
| 0.8785  | 1050  | 7.1094        | -               |
| 0.8994  | 1075  | 7.0538        | -               |
| 0.9203  | 1100  | 6.8486        | -               |
| 0.9412  | 1125  | 6.6782        | -               |
| 0.9621  | 1150  | 6.7144        | -               |
| 0.9831  | 1175  | 6.4156        | -               |
| 1.0     | 1196  | -             | 1.5533          |
| 1.0033  | 1200  | 6.1375        | -               |
| 1.0243  | 1225  | 6.1071        | -               |
| 1.0452  | 1250  | 6.1192        | -               |
| 1.0661  | 1275  | 6.0809        | -               |
| 1.0870  | 1300  | 5.9927        | -               |
| 1.1079  | 1325  | 5.9079        | -               |
| 1.1288  | 1350  | 5.7574        | -               |
| 1.1498  | 1375  | 5.5779        | -               |
| 1.1707  | 1400  | 5.6516        | -               |
| 1.1916  | 1425  | 5.685         | -               |
| 1.2125  | 1450  | 5.4679        | -               |
| 1.2334  | 1475  | 5.2991        | -               |
| 1.2543  | 1500  | 5.3907        | -               |
| 1.2753  | 1525  | 5.3169        | -               |
| 1.2962  | 1550  | 5.2292        | -               |
| 1.3171  | 1575  | 4.8167        | -               |
| 1.3380  | 1600  | 4.8537        | -               |
| 1.3589  | 1625  | 4.8508        | -               |
| 1.3798  | 1650  | 4.7141        | -               |
| 1.4008  | 1675  | 4.8972        | -               |
| 1.4217  | 1700  | 5.0055        | -               |
| 1.4426  | 1725  | 4.5549        | -               |
| 1.4635  | 1750  | 4.4827        | -               |
| 1.4844  | 1775  | 4.3768        | -               |
| 1.5053  | 1800  | 4.4797        | -               |
| 1.5262  | 1825  | 4.3233        | -               |
| 1.5472  | 1850  | 4.4144        | -               |
| 1.5681  | 1875  | 4.3645        | -               |
| 1.5890  | 1900  | 4.1553        | -               |
| 1.6099  | 1925  | 4.0135        | -               |
| 1.6308  | 1950  | 3.893         | -               |
| 1.6517  | 1975  | 3.8662        | -               |
| 1.6727  | 2000  | 3.7716        | -               |
| 1.6936  | 2025  | 3.6518        | -               |
| 1.7145  | 2050  | 3.8584        | -               |
| 1.7354  | 2075  | 3.6515        | -               |
| 1.7563  | 2100  | 3.7006        | -               |
| 1.7772  | 2125  | 3.5859        | -               |
| 1.7982  | 2150  | 3.517         | -               |
| 1.8191  | 2175  | 3.4801        | -               |
| 1.8400  | 2200  | 3.5071        | -               |
| 1.8609  | 2225  | 3.4548        | -               |
| 1.8818  | 2250  | 3.551         | -               |
| 1.9027  | 2275  | 3.2982        | -               |
| 1.9237  | 2300  | 3.2655        | -               |
| 1.9446  | 2325  | 3.299         | -               |
| 1.9655  | 2350  | 3.116         | -               |
| 1.9864  | 2375  | 3.2563        | -               |
| 2.0     | 2392  | -             | 0.7870          |
| 2.0067  | 2400  | 3.1901        | -               |
| 2.0276  | 2425  | 3.0254        | -               |
| 2.0485  | 2450  | 2.9005        | -               |
| 2.0694  | 2475  | 3.0706        | -               |
| 2.0904  | 2500  | 2.9535        | -               |
| 2.1113  | 2525  | 3.0661        | -               |
| 2.1322  | 2550  | 3.0015        | -               |
| 2.1531  | 2575  | 3.0453        | -               |
| 2.1740  | 2600  | 2.9803        | -               |
| 2.1949  | 2625  | 2.8875        | -               |
| 2.2159  | 2650  | 2.9146        | -               |
| 2.2368  | 2675  | 3.0393        | -               |
| 2.2577  | 2700  | 2.8208        | -               |
| 2.2786  | 2725  | 2.9839        | -               |
| 2.2995  | 2750  | 2.7769        | -               |
| 2.3204  | 2775  | 2.8205        | -               |
| 2.3414  | 2800  | 2.9002        | -               |
| 2.3623  | 2825  | 2.8541        | -               |
| 2.3832  | 2850  | 2.8854        | -               |
| 2.4041  | 2875  | 2.835         | -               |
| 2.4250  | 2900  | 2.856         | -               |
| 2.4459  | 2925  | 2.7811        | -               |
| 2.4668  | 2950  | 2.8198        | -               |
| 2.4878  | 2975  | 2.8754        | -               |
| 2.5087  | 3000  | 2.8293        | -               |
| 2.5296  | 3025  | 2.7363        | -               |
| 2.5505  | 3050  | 2.9321        | -               |
| 2.5714  | 3075  | 2.7308        | -               |
| 2.5923  | 3100  | 2.8159        | -               |
| 2.6133  | 3125  | 2.8787        | -               |
| 2.6342  | 3150  | 2.7792        | -               |
| 2.6551  | 3175  | 2.6451        | -               |
| 2.6760  | 3200  | 2.6023        | -               |
| 2.6969  | 3225  | 2.8164        | -               |
| 2.7178  | 3250  | 2.7252        | -               |
| 2.7388  | 3275  | 2.6332        | -               |
| 2.7597  | 3300  | 2.7831        | -               |
| 2.7806  | 3325  | 2.5779        | -               |
| 2.8015  | 3350  | 2.6805        | -               |
| 2.8224  | 3375  | 2.5948        | -               |
| 2.8433  | 3400  | 2.7419        | -               |
| 2.8643  | 3425  | 2.5551        | -               |
| 2.8852  | 3450  | 2.7116        | -               |
| 2.9061  | 3475  | 2.6787        | -               |
| 2.9270  | 3500  | 2.7321        | -               |
| 2.9479  | 3525  | 2.5877        | -               |
| 2.9688  | 3550  | 2.606         | -               |
| 2.9898  | 3575  | 2.5242        | -               |
| 3.0     | 3588  | -             | 0.6581          |
| 3.0100  | 3600  | 2.382         | -               |
| 3.0310  | 3625  | 2.3716        | -               |
| 3.0519  | 3650  | 2.4102        | -               |
| 3.0728  | 3675  | 2.4072        | -               |
| 3.0937  | 3700  | 2.5143        | -               |
| 3.1146  | 3725  | 2.4602        | -               |
| 3.1355  | 3750  | 2.408         | -               |
| 3.1565  | 3775  | 2.3126        | -               |
| 3.1774  | 3800  | 2.2041        | -               |
| 3.1983  | 3825  | 2.5298        | -               |
| 3.2192  | 3850  | 2.5002        | -               |
| 3.2401  | 3875  | 2.4565        | -               |
| 3.2610  | 3900  | 2.5427        | -               |
| 3.2819  | 3925  | 2.4844        | -               |
| 3.3029  | 3950  | 2.4338        | -               |
| 3.3238  | 3975  | 2.3609        | -               |
| 3.3447  | 4000  | 2.4852        | -               |
| 3.3656  | 4025  | 2.3381        | -               |
| 3.3865  | 4050  | 2.2397        | -               |
| 3.4074  | 4075  | 2.4058        | -               |
| 3.4284  | 4100  | 2.4348        | -               |
| 3.4493  | 4125  | 2.2591        | -               |
| 3.4702  | 4150  | 2.091         | -               |
| 3.4911  | 4175  | 2.2936        | -               |
| 3.5120  | 4200  | 2.1838        | -               |
| 3.5329  | 4225  | 2.2318        | -               |
| 3.5539  | 4250  | 2.314         | -               |
| 3.5748  | 4275  | 2.3764        | -               |
| 3.5957  | 4300  | 2.2741        | -               |
| 3.6166  | 4325  | 2.1413        | -               |
| 3.6375  | 4350  | 2.1878        | -               |
| 3.6584  | 4375  | 2.3597        | -               |
| 3.6794  | 4400  | 2.3349        | -               |
| 3.7003  | 4425  | 2.1769        | -               |
| 3.7212  | 4450  | 2.3394        | -               |
| 3.7421  | 4475  | 2.3737        | -               |
| 3.7630  | 4500  | 2.2395        | -               |
| 3.7839  | 4525  | 2.2581        | -               |
| 3.8049  | 4550  | 2.2883        | -               |
| 3.8258  | 4575  | 2.0617        | -               |
| 3.8467  | 4600  | 2.0931        | -               |
| 3.8676  | 4625  | 2.1205        | -               |
| 3.8885  | 4650  | 2.0244        | -               |
| 3.9094  | 4675  | 2.1658        | -               |
| 3.9303  | 4700  | 2.1116        | -               |
| 3.9513  | 4725  | 2.1416        | -               |
| 3.9722  | 4750  | 2.1715        | -               |
| 3.9931  | 4775  | 2.1152        | -               |
| 4.0     | 4784  | -             | 0.5600          |
| 4.0134  | 4800  | 1.8343        | -               |
| 4.0343  | 4825  | 1.8235        | -               |
| 4.0552  | 4850  | 2.143         | -               |
| 4.0761  | 4875  | 1.8487        | -               |
| 4.0971  | 4900  | 1.9212        | -               |
| 4.1180  | 4925  | 1.8809        | -               |
| 4.1389  | 4950  | 1.859         | -               |
| 4.1598  | 4975  | 1.8767        | -               |
| 4.1807  | 5000  | 1.9545        | -               |
| 4.2016  | 5025  | 1.7631        | -               |
| 4.2225  | 5050  | 1.9293        | -               |
| 4.2435  | 5075  | 1.8834        | -               |
| 4.2644  | 5100  | 1.913         | -               |
| 4.2853  | 5125  | 1.7772        | -               |
| 4.3062  | 5150  | 1.9589        | -               |
| 4.3271  | 5175  | 1.9194        | -               |
| 4.3480  | 5200  | 1.9532        | -               |
| 4.3690  | 5225  | 1.8412        | -               |
| 4.3899  | 5250  | 1.7999        | -               |
| 4.4108  | 5275  | 1.8209        | -               |
| 4.4317  | 5300  | 1.8233        | -               |
| 4.4526  | 5325  | 2.0254        | -               |
| 4.4735  | 5350  | 1.7253        | -               |
| 4.4945  | 5375  | 1.8376        | -               |
| 4.5154  | 5400  | 1.9594        | -               |
| 4.5363  | 5425  | 1.9021        | -               |
| 4.5572  | 5450  | 1.9151        | -               |
| 4.5781  | 5475  | 1.6769        | -               |
| 4.5990  | 5500  | 1.8645        | -               |
| 4.6200  | 5525  | 1.8434        | -               |
| 4.6409  | 5550  | 1.8237        | -               |
| 4.6618  | 5575  | 1.857         | -               |
| 4.6827  | 5600  | 1.7337        | -               |
| 4.7036  | 5625  | 1.7153        | -               |
| 4.7245  | 5650  | 1.8292        | -               |
| 4.7455  | 5675  | 1.6642        | -               |
| 4.7664  | 5700  | 1.8444        | -               |
| 4.7873  | 5725  | 1.9442        | -               |
| 4.8082  | 5750  | 1.8337        | -               |
| 4.8291  | 5775  | 1.7749        | -               |
| 4.8500  | 5800  | 1.7826        | -               |
| 4.8709  | 5825  | 1.8249        | -               |
| 4.8919  | 5850  | 1.8178        | -               |
| 4.9128  | 5875  | 1.7442        | -               |
| 4.9337  | 5900  | 1.7301        | -               |
| 4.9546  | 5925  | 1.7219        | -               |
| 4.9755  | 5950  | 1.6625        | -               |
| 4.9964  | 5975  | 1.7832        | -               |
| 5.0     | 5980  | -             | 0.5572          |
| 5.0167  | 6000  | 1.6186        | -               |
| 5.0376  | 6025  | 1.7149        | -               |
| 5.0586  | 6050  | 1.5507        | -               |
| 5.0795  | 6075  | 1.6611        | -               |
| 5.1004  | 6100  | 1.7019        | -               |
| 5.1213  | 6125  | 1.6496        | -               |
| 5.1422  | 6150  | 1.6236        | -               |
| 5.1631  | 6175  | 1.6661        | -               |
| 5.1841  | 6200  | 1.6319        | -               |
| 5.2050  | 6225  | 1.4661        | -               |
| 5.2259  | 6250  | 1.5992        | -               |
| 5.2468  | 6275  | 1.6284        | -               |
| 5.2677  | 6300  | 1.701         | -               |
| 5.2886  | 6325  | 1.608         | -               |
| 5.3096  | 6350  | 1.503         | -               |
| 5.3305  | 6375  | 1.5214        | -               |
| 5.3514  | 6400  | 1.6017        | -               |
| 5.3723  | 6425  | 1.4845        | -               |
| 5.3932  | 6450  | 1.536         | -               |
| 5.4141  | 6475  | 1.5897        | -               |
| 5.4351  | 6500  | 1.5673        | -               |
| 5.4560  | 6525  | 1.4894        | -               |
| 5.4769  | 6550  | 1.5885        | -               |
| 5.4978  | 6575  | 1.5837        | -               |
| 5.5187  | 6600  | 1.5758        | -               |
| 5.5396  | 6625  | 1.5556        | -               |
| 5.5606  | 6650  | 1.6003        | -               |
| 5.5815  | 6675  | 1.4955        | -               |
| 5.6024  | 6700  | 1.5281        | -               |
| 5.6233  | 6725  | 1.5394        | -               |
| 5.6442  | 6750  | 1.5364        | -               |
| 5.6651  | 6775  | 1.6196        | -               |
| 5.6860  | 6800  | 1.6091        | -               |
| 5.7070  | 6825  | 1.4852        | -               |
| 5.7279  | 6850  | 1.5032        | -               |
| 5.7488  | 6875  | 1.6116        | -               |
| 5.7697  | 6900  | 1.6532        | -               |
| 5.7906  | 6925  | 1.5542        | -               |
| 5.8115  | 6950  | 1.6638        | -               |
| 5.8325  | 6975  | 1.6459        | -               |
| 5.8534  | 7000  | 1.5373        | -               |
| 5.8743  | 7025  | 1.5356        | -               |
| 5.8952  | 7050  | 1.6021        | -               |
| 5.9161  | 7075  | 1.4595        | -               |
| 5.9370  | 7100  | 1.6702        | -               |
| 5.9580  | 7125  | 1.5389        | -               |
| 5.9789  | 7150  | 1.6248        | -               |
| 5.9998  | 7175  | 1.4984        | -               |
| 6.0     | 7176  | -             | 0.5100          |
| 6.0201  | 7200  | 1.3695        | -               |
| 6.0410  | 7225  | 1.3322        | -               |
| 6.0619  | 7250  | 1.4368        | -               |
| 6.0828  | 7275  | 1.4408        | -               |
| 6.1037  | 7300  | 1.3616        | -               |
| 6.1247  | 7325  | 1.4339        | -               |
| 6.1456  | 7350  | 1.3903        | -               |
| 6.1665  | 7375  | 1.3642        | -               |
| 6.1874  | 7400  | 1.4748        | -               |
| 6.2083  | 7425  | 1.3995        | -               |
| 6.2292  | 7450  | 1.4648        | -               |
| 6.2502  | 7475  | 1.3695        | -               |
| 6.2711  | 7500  | 1.4351        | -               |
| 6.2920  | 7525  | 1.2769        | -               |
| 6.3129  | 7550  | 1.3428        | -               |
| 6.3338  | 7575  | 1.4283        | -               |
| 6.3547  | 7600  | 1.3561        | -               |
| 6.3757  | 7625  | 1.3511        | -               |
| 6.3966  | 7650  | 1.269         | -               |
| 6.4175  | 7675  | 1.397         | -               |
| 6.4384  | 7700  | 1.3518        | -               |
| 6.4593  | 7725  | 1.4609        | -               |
| 6.4802  | 7750  | 1.4078        | -               |
| 6.5012  | 7775  | 1.402         | -               |
| 6.5221  | 7800  | 1.5066        | -               |
| 6.5430  | 7825  | 1.4362        | -               |
| 6.5639  | 7850  | 1.3338        | -               |
| 6.5848  | 7875  | 1.3674        | -               |
| 6.6057  | 7900  | 1.5026        | -               |
| 6.6266  | 7925  | 1.4372        | -               |
| 6.6476  | 7950  | 1.4082        | -               |
| 6.6685  | 7975  | 1.4072        | -               |
| 6.6894  | 8000  | 1.3574        | -               |
| 6.7103  | 8025  | 1.4536        | -               |
| 6.7312  | 8050  | 1.3384        | -               |
| 6.7521  | 8075  | 1.359         | -               |
| 6.7731  | 8100  | 1.345         | -               |
| 6.7940  | 8125  | 1.4538        | -               |
| 6.8149  | 8150  | 1.4173        | -               |
| 6.8358  | 8175  | 1.3518        | -               |
| 6.8567  | 8200  | 1.5022        | -               |
| 6.8776  | 8225  | 1.3478        | -               |
| 6.8986  | 8250  | 1.3726        | -               |
| 6.9195  | 8275  | 1.3032        | -               |
| 6.9404  | 8300  | 1.4186        | -               |
| 6.9613  | 8325  | 1.3607        | -               |
| 6.9822  | 8350  | 1.3865        | -               |
| 7.0     | 8372  | -             | 0.5318          |
| 7.0025  | 8375  | 1.3294        | -               |
| 7.0234  | 8400  | 1.2017        | -               |
| 7.0443  | 8425  | 1.3195        | -               |
| 7.0653  | 8450  | 1.3368        | -               |
| 7.0862  | 8475  | 1.1384        | -               |
| 7.1071  | 8500  | 1.1681        | -               |
| 7.1280  | 8525  | 1.2337        | -               |
| 7.1489  | 8550  | 1.2014        | -               |
| 7.1698  | 8575  | 1.2172        | -               |
| 7.1908  | 8600  | 1.1622        | -               |
| 7.2117  | 8625  | 1.28          | -               |
| 7.2326  | 8650  | 1.1788        | -               |
| 7.2535  | 8675  | 1.2071        | -               |
| 7.2744  | 8700  | 1.3058        | -               |
| 7.2953  | 8725  | 1.0641        | -               |
| 7.3163  | 8750  | 1.2148        | -               |
| 7.3372  | 8775  | 1.2733        | -               |
| 7.3581  | 8800  | 1.3021        | -               |
| 7.3790  | 8825  | 1.2135        | -               |
| 7.3999  | 8850  | 1.2448        | -               |
| 7.4208  | 8875  | 1.2395        | -               |
| 7.4417  | 8900  | 1.2092        | -               |
| 7.4627  | 8925  | 1.2014        | -               |
| 7.4836  | 8950  | 1.2796        | -               |
| 7.5045  | 8975  | 1.1799        | -               |
| 7.5254  | 9000  | 1.2803        | -               |
| 7.5463  | 9025  | 1.2035        | -               |
| 7.5672  | 9050  | 1.322         | -               |
| 7.5882  | 9075  | 1.1686        | -               |
| 7.6091  | 9100  | 1.2026        | -               |
| 7.6300  | 9125  | 1.166         | -               |
| 7.6509  | 9150  | 1.1726        | -               |
| 7.6718  | 9175  | 1.1399        | -               |
| 7.6927  | 9200  | 1.247         | -               |
| 7.7137  | 9225  | 1.1745        | -               |
| 7.7346  | 9250  | 1.2009        | -               |
| 7.7555  | 9275  | 1.1583        | -               |
| 7.7764  | 9300  | 1.1602        | -               |
| 7.7973  | 9325  | 1.1662        | -               |
| 7.8182  | 9350  | 1.1921        | -               |
| 7.8392  | 9375  | 1.3283        | -               |
| 7.8601  | 9400  | 1.2297        | -               |
| 7.8810  | 9425  | 1.1741        | -               |
| 7.9019  | 9450  | 1.1693        | -               |
| 7.9228  | 9475  | 1.2106        | -               |
| 7.9437  | 9500  | 1.2158        | -               |
| 7.9647  | 9525  | 1.2278        | -               |
| 7.9856  | 9550  | 1.1588        | -               |
| 8.0     | 9568  | -             | 0.4836          |
| 8.0059  | 9575  | 1.0466        | -               |
| 8.0268  | 9600  | 1.0246        | -               |
| 8.0477  | 9625  | 1.0779        | -               |
| 8.0686  | 9650  | 1.108         | -               |
| 8.0895  | 9675  | 1.1298        | -               |
| 8.1104  | 9700  | 0.9884        | -               |
| 8.1314  | 9725  | 1.1435        | -               |
| 8.1523  | 9750  | 1.0603        | -               |
| 8.1732  | 9775  | 1.0275        | -               |
| 8.1941  | 9800  | 1.0149        | -               |
| 8.2150  | 9825  | 1.1173        | -               |
| 8.2359  | 9850  | 1.0842        | -               |
| 8.2569  | 9875  | 1.1222        | -               |
| 8.2778  | 9900  | 1.1037        | -               |
| 8.2987  | 9925  | 1.1466        | -               |
| 8.3196  | 9950  | 1.1432        | -               |
| 8.3405  | 9975  | 1.1033        | -               |
| 8.3614  | 10000 | 1.1009        | -               |
| 8.3823  | 10025 | 1.0739        | -               |
| 8.4033  | 10050 | 1.1836        | -               |
| 8.4242  | 10075 | 1.0854        | -               |
| 8.4451  | 10100 | 1.0688        | -               |
| 8.4660  | 10125 | 1.1397        | -               |
| 8.4869  | 10150 | 1.0196        | -               |
| 8.5078  | 10175 | 1.0563        | -               |
| 8.5288  | 10200 | 1.0692        | -               |
| 8.5497  | 10225 | 1.1298        | -               |
| 8.5706  | 10250 | 1.0564        | -               |
| 8.5915  | 10275 | 1.0236        | -               |
| 8.6124  | 10300 | 1.0193        | -               |
| 8.6333  | 10325 | 1.146         | -               |
| 8.6543  | 10350 | 1.0225        | -               |
| 8.6752  | 10375 | 1.1176        | -               |
| 8.6961  | 10400 | 1.1484        | -               |
| 8.7170  | 10425 | 1.0531        | -               |
| 8.7379  | 10450 | 1.1234        | -               |
| 8.7588  | 10475 | 1.0093        | -               |
| 8.7798  | 10500 | 1.1289        | -               |
| 8.8007  | 10525 | 1.0553        | -               |
| 8.8216  | 10550 | 1.051         | -               |
| 8.8425  | 10575 | 1.1768        | -               |
| 8.8634  | 10600 | 1.1101        | -               |
| 8.8843  | 10625 | 1.0749        | -               |
| 8.9052  | 10650 | 1.065         | -               |
| 8.9262  | 10675 | 0.9639        | -               |
| 8.9471  | 10700 | 1.1494        | -               |
| 8.9680  | 10725 | 1.0733        | -               |
| 8.9889  | 10750 | 1.0315        | -               |
| 9.0     | 10764 | -             | 0.4749          |
| 9.0092  | 10775 | 0.9844        | -               |
| 9.0301  | 10800 | 0.9321        | -               |
| 9.0510  | 10825 | 0.9686        | -               |
| 9.0720  | 10850 | 0.8765        | -               |
| 9.0929  | 10875 | 0.9307        | -               |
| 9.1138  | 10900 | 0.9941        | -               |
| 9.1347  | 10925 | 1.0016        | -               |
| 9.1556  | 10950 | 0.9614        | -               |
| 9.1765  | 10975 | 0.9606        | -               |
| 9.1974  | 11000 | 0.9288        | -               |
| 9.2184  | 11025 | 0.9787        | -               |
| 9.2393  | 11050 | 0.9742        | -               |
| 9.2602  | 11075 | 0.9023        | -               |
| 9.2811  | 11100 | 0.9229        | -               |
| 9.3020  | 11125 | 1.0158        | -               |
| 9.3229  | 11150 | 0.9306        | -               |
| 9.3439  | 11175 | 0.8686        | -               |
| 9.3648  | 11200 | 0.9462        | -               |
| 9.3857  | 11225 | 0.982         | -               |
| 9.4066  | 11250 | 0.9382        | -               |
| 9.4275  | 11275 | 0.9802        | -               |
| 9.4484  | 11300 | 0.952         | -               |
| 9.4694  | 11325 | 0.9896        | -               |
| 9.4903  | 11350 | 0.9342        | -               |
| 9.5112  | 11375 | 0.9805        | -               |
| 9.5321  | 11400 | 1.0328        | -               |
| 9.5530  | 11425 | 1.0005        | -               |
| 9.5739  | 11450 | 1.0205        | -               |
| 9.5949  | 11475 | 0.9793        | -               |
| 9.6158  | 11500 | 0.9898        | -               |
| 9.6367  | 11525 | 0.8843        | -               |
| 9.6576  | 11550 | 1.0141        | -               |
| 9.6785  | 11575 | 0.9124        | -               |
| 9.6994  | 11600 | 0.913         | -               |
| 9.7204  | 11625 | 0.9523        | -               |
| 9.7413  | 11650 | 0.9536        | -               |
| 9.7622  | 11675 | 0.9891        | -               |
| 9.7831  | 11700 | 1.009         | -               |
| 9.8040  | 11725 | 0.9993        | -               |
| 9.8249  | 11750 | 0.993         | -               |
| 9.8458  | 11775 | 1.0415        | -               |
| 9.8668  | 11800 | 0.9055        | -               |
| 9.8877  | 11825 | 0.9556        | -               |
| 9.9086  | 11850 | 0.9079        | -               |
| 9.9295  | 11875 | 0.9678        | -               |
| 9.9504  | 11900 | 0.9804        | -               |
| 9.9713  | 11925 | 0.9406        | -               |
| 9.9923  | 11950 | 1.0289        | -               |
| 10.0    | 11960 | -             | 0.4548          |
| 10.0125 | 11975 | 0.9349        | -               |
| 10.0335 | 12000 | 0.8442        | -               |
| 10.0544 | 12025 | 0.9024        | -               |
| 10.0753 | 12050 | 0.9275        | -               |
| 10.0962 | 12075 | 0.8773        | -               |
| 10.1171 | 12100 | 0.9343        | -               |
| 10.1380 | 12125 | 0.8364        | -               |
| 10.1590 | 12150 | 0.9005        | -               |
| 10.1799 | 12175 | 0.8115        | -               |
| 10.2008 | 12200 | 0.8444        | -               |
| 10.2217 | 12225 | 0.9078        | -               |
| 10.2426 | 12250 | 0.9739        | -               |
| 10.2635 | 12275 | 0.8104        | -               |
| 10.2845 | 12300 | 0.8658        | -               |
| 10.3054 | 12325 | 0.8037        | -               |
| 10.3263 | 12350 | 0.9003        | -               |
| 10.3472 | 12375 | 0.8337        | -               |
| 10.3681 | 12400 | 0.9418        | -               |
| 10.3890 | 12425 | 0.9715        | -               |
| 10.4100 | 12450 | 0.8675        | -               |
| 10.4309 | 12475 | 0.948         | -               |
| 10.4518 | 12500 | 0.8591        | -               |
| 10.4727 | 12525 | 0.8112        | -               |
| 10.4936 | 12550 | 0.8664        | -               |
| 10.5145 | 12575 | 0.9338        | -               |
| 10.5355 | 12600 | 0.9484        | -               |
| 10.5564 | 12625 | 0.8962        | -               |
| 10.5773 | 12650 | 0.885         | -               |
| 10.5982 | 12675 | 0.8738        | -               |
| 10.6191 | 12700 | 0.8757        | -               |
| 10.6400 | 12725 | 0.9193        | -               |
| 10.6609 | 12750 | 0.8981        | -               |
| 10.6819 | 12775 | 0.8541        | -               |
| 10.7028 | 12800 | 0.8221        | -               |
| 10.7237 | 12825 | 0.8744        | -               |
| 10.7446 | 12850 | 0.9104        | -               |
| 10.7655 | 12875 | 0.7979        | -               |
| 10.7864 | 12900 | 0.9228        | -               |
| 10.8074 | 12925 | 0.966         | -               |
| 10.8283 | 12950 | 0.9986        | -               |
| 10.8492 | 12975 | 0.8476        | -               |
| 10.8701 | 13000 | 0.8677        | -               |
| 10.8910 | 13025 | 0.8762        | -               |
| 10.9119 | 13050 | 0.7595        | -               |
| 10.9329 | 13075 | 0.8689        | -               |
| 10.9538 | 13100 | 0.8865        | -               |
| 10.9747 | 13125 | 0.8458        | -               |
| 10.9956 | 13150 | 0.8227        | -               |
| 11.0    | 13156 | -             | 0.4538          |
| 11.0159 | 13175 | 0.7391        | -               |
| 11.0368 | 13200 | 0.8541        | -               |
| 11.0577 | 13225 | 0.8224        | -               |
| 11.0786 | 13250 | 0.8527        | -               |
| 11.0996 | 13275 | 0.7857        | -               |
| 11.1205 | 13300 | 0.8313        | -               |
| 11.1414 | 13325 | 0.8267        | -               |
| 11.1623 | 13350 | 0.77          | -               |
| 11.1832 | 13375 | 0.8348        | -               |
| 11.2041 | 13400 | 0.8489        | -               |
| 11.2251 | 13425 | 0.8131        | -               |
| 11.2460 | 13450 | 0.8189        | -               |
| 11.2669 | 13475 | 0.8243        | -               |
| 11.2878 | 13500 | 0.7921        | -               |
| 11.3087 | 13525 | 0.855         | -               |
| 11.3296 | 13550 | 0.8256        | -               |
| 11.3506 | 13575 | 0.8343        | -               |
| 11.3715 | 13600 | 0.756         | -               |
| 11.3924 | 13625 | 0.7864        | -               |
| 11.4133 | 13650 | 0.7711        | -               |
| 11.4342 | 13675 | 0.7796        | -               |
| 11.4551 | 13700 | 0.7453        | -               |
| 11.4761 | 13725 | 0.8036        | -               |
| 11.4970 | 13750 | 0.8103        | -               |
| 11.5179 | 13775 | 0.7875        | -               |
| 11.5388 | 13800 | 0.8129        | -               |
| 11.5597 | 13825 | 0.7948        | -               |
| 11.5806 | 13850 | 0.7964        | -               |
| 11.6015 | 13875 | 0.7779        | -               |
| 11.6225 | 13900 | 0.8822        | -               |
| 11.6434 | 13925 | 0.816         | -               |
| 11.6643 | 13950 | 0.8655        | -               |
| 11.6852 | 13975 | 0.8618        | -               |
| 11.7061 | 14000 | 0.7549        | -               |
| 11.7270 | 14025 | 0.7753        | -               |
| 11.7480 | 14050 | 0.8333        | -               |
| 11.7689 | 14075 | 0.874         | -               |
| 11.7898 | 14100 | 0.8405        | -               |
| 11.8107 | 14125 | 0.7769        | -               |
| 11.8316 | 14150 | 0.9512        | -               |
| 11.8525 | 14175 | 0.8139        | -               |
| 11.8735 | 14200 | 0.8443        | -               |
| 11.8944 | 14225 | 0.7202        | -               |
| 11.9153 | 14250 | 0.824         | -               |
| 11.9362 | 14275 | 0.8253        | -               |
| 11.9571 | 14300 | 0.8774        | -               |
| 11.9780 | 14325 | 0.8846        | -               |
| 11.9990 | 14350 | 0.8147        | -               |
| 12.0    | 14352 | -             | 0.4423          |
| 12.0192 | 14375 | 0.6773        | -               |
| 12.0402 | 14400 | 0.6991        | -               |
| 12.0611 | 14425 | 0.7091        | -               |
| 12.0820 | 14450 | 0.7756        | -               |
| 12.1029 | 14475 | 0.7402        | -               |
| 12.1238 | 14500 | 0.747         | -               |
| 12.1447 | 14525 | 0.7846        | -               |
| 12.1657 | 14550 | 0.6709        | -               |
| 12.1866 | 14575 | 0.7079        | -               |
| 12.2075 | 14600 | 0.6596        | -               |
| 12.2284 | 14625 | 0.7578        | -               |
| 12.2493 | 14650 | 0.7966        | -               |
| 12.2702 | 14675 | 0.746         | -               |
| 12.2912 | 14700 | 0.7483        | -               |
| 12.3121 | 14725 | 0.7809        | -               |
| 12.3330 | 14750 | 0.8817        | -               |
| 12.3539 | 14775 | 0.8136        | -               |
| 12.3748 | 14800 | 0.7599        | -               |
| 12.3957 | 14825 | 0.7058        | -               |
| 12.4166 | 14850 | 0.7921        | -               |
| 12.4376 | 14875 | 0.7388        | -               |
| 12.4585 | 14900 | 0.8223        | -               |
| 12.4794 | 14925 | 0.7876        | -               |
| 12.5003 | 14950 | 0.7929        | -               |
| 12.5212 | 14975 | 0.7853        | -               |
| 12.5421 | 15000 | 0.8239        | -               |
| 12.5631 | 15025 | 0.6847        | -               |
| 12.5840 | 15050 | 0.8068        | -               |
| 12.6049 | 15075 | 0.7443        | -               |
| 12.6258 | 15100 | 0.6757        | -               |
| 12.6467 | 15125 | 0.7256        | -               |
| 12.6676 | 15150 | 0.7592        | -               |
| 12.6886 | 15175 | 0.7707        | -               |
| 12.7095 | 15200 | 0.7829        | -               |
| 12.7304 | 15225 | 0.7196        | -               |
| 12.7513 | 15250 | 0.7939        | -               |
| 12.7722 | 15275 | 0.7345        | -               |
| 12.7931 | 15300 | 0.7611        | -               |
| 12.8141 | 15325 | 0.7725        | -               |
| 12.8350 | 15350 | 0.7241        | -               |
| 12.8559 | 15375 | 0.7609        | -               |
| 12.8768 | 15400 | 0.7884        | -               |
| 12.8977 | 15425 | 0.7171        | -               |
| 12.9186 | 15450 | 0.6882        | -               |
| 12.9396 | 15475 | 0.7696        | -               |
| 12.9605 | 15500 | 0.7801        | -               |
| 12.9814 | 15525 | 0.7752        | -               |
| 13.0    | 15548 | -             | 0.4610          |
| 13.0017 | 15550 | 0.6396        | -               |
| 13.0226 | 15575 | 0.7092        | -               |
| 13.0435 | 15600 | 0.7368        | -               |
| 13.0644 | 15625 | 0.7475        | -               |
| 13.0853 | 15650 | 0.6743        | -               |
| 13.1063 | 15675 | 0.7116        | -               |
| 13.1272 | 15700 | 0.7462        | -               |
| 13.1481 | 15725 | 0.6763        | -               |
| 13.1690 | 15750 | 0.7293        | -               |
| 13.1899 | 15775 | 0.6434        | -               |
| 13.2108 | 15800 | 0.6198        | -               |
| 13.2318 | 15825 | 0.7185        | -               |
| 13.2527 | 15850 | 0.7018        | -               |
| 13.2736 | 15875 | 0.6736        | -               |
| 13.2945 | 15900 | 0.697         | -               |
| 13.3154 | 15925 | 0.6995        | -               |
| 13.3363 | 15950 | 0.723         | -               |
| 13.3572 | 15975 | 0.6622        | -               |
| 13.3782 | 16000 | 0.6897        | -               |
| 13.3991 | 16025 | 0.6385        | -               |
| 13.4200 | 16050 | 0.6531        | -               |
| 13.4409 | 16075 | 0.6754        | -               |
| 13.4618 | 16100 | 0.74          | -               |
| 13.4827 | 16125 | 0.653         | -               |
| 13.5037 | 16150 | 0.692         | -               |
| 13.5246 | 16175 | 0.7085        | -               |
| 13.5455 | 16200 | 0.7456        | -               |
| 13.5664 | 16225 | 0.7109        | -               |
| 13.5873 | 16250 | 0.7132        | -               |
| 13.6082 | 16275 | 0.6414        | -               |
| 13.6292 | 16300 | 0.6783        | -               |
| 13.6501 | 16325 | 0.6649        | -               |
| 13.6710 | 16350 | 0.6865        | -               |
| 13.6919 | 16375 | 0.6906        | -               |
| 13.7128 | 16400 | 0.7297        | -               |
| 13.7337 | 16425 | 0.7661        | -               |
| 13.7547 | 16450 | 0.6821        | -               |
| 13.7756 | 16475 | 0.7291        | -               |
| 13.7965 | 16500 | 0.6903        | -               |
| 13.8174 | 16525 | 0.79          | -               |
| 13.8383 | 16550 | 0.6433        | -               |
| 13.8592 | 16575 | 0.8043        | -               |
| 13.8802 | 16600 | 0.6494        | -               |
| 13.9011 | 16625 | 0.7725        | -               |
| 13.9220 | 16650 | 0.6874        | -               |
| 13.9429 | 16675 | 0.6188        | -               |
| 13.9638 | 16700 | 0.6982        | -               |
| 13.9847 | 16725 | 0.7526        | -               |
| 14.0    | 16744 | -             | 0.4506          |
| 14.0050 | 16750 | 0.6476        | -               |
| 14.0259 | 16775 | 0.7261        | -               |
| 14.0469 | 16800 | 0.6569        | -               |
| 14.0678 | 16825 | 0.6602        | -               |
| 14.0887 | 16850 | 0.5839        | -               |
| 14.1096 | 16875 | 0.579         | -               |
| 14.1305 | 16900 | 0.6535        | -               |
| 14.1514 | 16925 | 0.6594        | -               |
| 14.1723 | 16950 | 0.6087        | -               |
| 14.1933 | 16975 | 0.6623        | -               |
| 14.2142 | 17000 | 0.6831        | -               |
| 14.2351 | 17025 | 0.6431        | -               |
| 14.2560 | 17050 | 0.6878        | -               |
| 14.2769 | 17075 | 0.5833        | -               |
| 14.2978 | 17100 | 0.6051        | -               |
| 14.3188 | 17125 | 0.6046        | -               |
| 14.3397 | 17150 | 0.6972        | -               |
| 14.3606 | 17175 | 0.653         | -               |
| 14.3815 | 17200 | 0.6017        | -               |
| 14.4024 | 17225 | 0.6916        | -               |
| 14.4233 | 17250 | 0.6885        | -               |
| 14.4443 | 17275 | 0.6409        | -               |
| 14.4652 | 17300 | 0.646         | -               |
| 14.4861 | 17325 | 0.6741        | -               |
| 14.5070 | 17350 | 0.6596        | -               |
| 14.5279 | 17375 | 0.591         | -               |
| 14.5488 | 17400 | 0.6842        | -               |
| 14.5698 | 17425 | 0.6819        | -               |
| 14.5907 | 17450 | 0.7014        | -               |
| 14.6116 | 17475 | 0.6357        | -               |
| 14.6325 | 17500 | 0.6292        | -               |
| 14.6534 | 17525 | 0.7117        | -               |
| 14.6743 | 17550 | 0.661         | -               |
| 14.6953 | 17575 | 0.6645        | -               |
| 14.7162 | 17600 | 0.6753        | -               |
| 14.7371 | 17625 | 0.6701        | -               |
| 14.7580 | 17650 | 0.6218        | -               |
| 14.7789 | 17675 | 0.6901        | -               |
| 14.7998 | 17700 | 0.6762        | -               |
| 14.8207 | 17725 | 0.7379        | -               |
| 14.8417 | 17750 | 0.6016        | -               |
| 14.8626 | 17775 | 0.7788        | -               |
| 14.8835 | 17800 | 0.6845        | -               |
| 14.9044 | 17825 | 0.6252        | -               |
| 14.9253 | 17850 | 0.7042        | -               |
| 14.9462 | 17875 | 0.6442        | -               |
| 14.9672 | 17900 | 0.6373        | -               |
| 14.9881 | 17925 | 0.5784        | -               |
| 15.0    | 17940 | -             | 0.4562          |
| 15.0084 | 17950 | 0.5881        | -               |
| 15.0293 | 17975 | 0.5685        | -               |
| 15.0502 | 18000 | 0.582         | -               |
| 15.0711 | 18025 | 0.589         | -               |
| 15.0920 | 18050 | 0.6258        | -               |
| 15.1129 | 18075 | 0.563         | -               |
| 15.1339 | 18100 | 0.729         | -               |
| 15.1548 | 18125 | 0.58          | -               |
| 15.1757 | 18150 | 0.6685        | -               |
| 15.1966 | 18175 | 0.5521        | -               |
| 15.2175 | 18200 | 0.5728        | -               |
| 15.2384 | 18225 | 0.6593        | -               |
| 15.2594 | 18250 | 0.5694        | -               |
| 15.2803 | 18275 | 0.5822        | -               |
| 15.3012 | 18300 | 0.6531        | -               |
| 15.3221 | 18325 | 0.6675        | -               |
| 15.3430 | 18350 | 0.5966        | -               |
| 15.3639 | 18375 | 0.58          | -               |
| 15.3849 | 18400 | 0.6075        | -               |
| 15.4058 | 18425 | 0.6257        | -               |
| 15.4267 | 18450 | 0.6292        | -               |
| 15.4476 | 18475 | 0.5595        | -               |
| 15.4685 | 18500 | 0.6655        | -               |
| 15.4894 | 18525 | 0.6413        | -               |
| 15.5104 | 18550 | 0.6069        | -               |
| 15.5313 | 18575 | 0.6854        | -               |
| 15.5522 | 18600 | 0.5882        | -               |
| 15.5731 | 18625 | 0.5884        | -               |
| 15.5940 | 18650 | 0.6388        | -               |
| 15.6149 | 18675 | 0.6495        | -               |
| 15.6359 | 18700 | 0.6271        | -               |
| 15.6568 | 18725 | 0.5517        | -               |
| 15.6777 | 18750 | 0.5516        | -               |
| 15.6986 | 18775 | 0.5962        | -               |
| 15.7195 | 18800 | 0.6247        | -               |
| 15.7404 | 18825 | 0.5642        | -               |
| 15.7613 | 18850 | 0.6036        | -               |
| 15.7823 | 18875 | 0.548         | -               |
| 15.8032 | 18900 | 0.6763        | -               |
| 15.8241 | 18925 | 0.6853        | -               |
| 15.8450 | 18950 | 0.64          | -               |
| 15.8659 | 18975 | 0.6034        | -               |
| 15.8868 | 19000 | 0.564         | -               |
| 15.9078 | 19025 | 0.5936        | -               |
| 15.9287 | 19050 | 0.5937        | -               |
| 15.9496 | 19075 | 0.6297        | -               |
| 15.9705 | 19100 | 0.6943        | -               |
| 15.9914 | 19125 | 0.5509        | -               |
| 16.0    | 19136 | -             | 0.4354          |
| 16.0117 | 19150 | 0.5487        | -               |
| 16.0326 | 19175 | 0.5844        | -               |
| 16.0535 | 19200 | 0.5832        | -               |
| 16.0745 | 19225 | 0.5547        | -               |
| 16.0954 | 19250 | 0.5673        | -               |
| 16.1163 | 19275 | 0.601         | -               |
| 16.1372 | 19300 | 0.6294        | -               |
| 16.1581 | 19325 | 0.5535        | -               |
| 16.1790 | 19350 | 0.6101        | -               |
| 16.2000 | 19375 | 0.5296        | -               |
| 16.2209 | 19400 | 0.5613        | -               |
| 16.2418 | 19425 | 0.5873        | -               |
| 16.2627 | 19450 | 0.5674        | -               |
| 16.2836 | 19475 | 0.5287        | -               |
| 16.3045 | 19500 | 0.6249        | -               |
| 16.3255 | 19525 | 0.5603        | -               |
| 16.3464 | 19550 | 0.5431        | -               |
| 16.3673 | 19575 | 0.6321        | -               |
| 16.3882 | 19600 | 0.6021        | -               |
| 16.4091 | 19625 | 0.5824        | -               |
| 16.4300 | 19650 | 0.5851        | -               |
| 16.4510 | 19675 | 0.5976        | -               |
| 16.4719 | 19700 | 0.5936        | -               |
| 16.4928 | 19725 | 0.6091        | -               |
| 16.5137 | 19750 | 0.5668        | -               |
| 16.5346 | 19775 | 0.5864        | -               |
| 16.5555 | 19800 | 0.5877        | -               |
| 16.5764 | 19825 | 0.5256        | -               |
| 16.5974 | 19850 | 0.6262        | -               |
| 16.6183 | 19875 | 0.544         | -               |
| 16.6392 | 19900 | 0.6014        | -               |
| 16.6601 | 19925 | 0.5266        | -               |
| 16.6810 | 19950 | 0.6296        | -               |
| 16.7019 | 19975 | 0.5509        | -               |
| 16.7229 | 20000 | 0.5638        | -               |
| 16.7438 | 20025 | 0.5723        | -               |
| 16.7647 | 20050 | 0.6048        | -               |
| 16.7856 | 20075 | 0.584         | -               |
| 16.8065 | 20100 | 0.6025        | -               |
| 16.8274 | 20125 | 0.5264        | -               |
| 16.8484 | 20150 | 0.5268        | -               |
| 16.8693 | 20175 | 0.6272        | -               |
| 16.8902 | 20200 | 0.5727        | -               |
| 16.9111 | 20225 | 0.5847        | -               |
| 16.9320 | 20250 | 0.5681        | -               |
| 16.9529 | 20275 | 0.6079        | -               |
| 16.9739 | 20300 | 0.5505        | -               |
| 16.9948 | 20325 | 0.5985        | -               |
| 17.0    | 20332 | -             | 0.4529          |
| 17.0151 | 20350 | 0.5469        | -               |
| 17.0360 | 20375 | 0.5448        | -               |
| 17.0569 | 20400 | 0.5774        | -               |
| 17.0778 | 20425 | 0.5858        | -               |
| 17.0987 | 20450 | 0.5509        | -               |
| 17.1196 | 20475 | 0.5456        | -               |
| 17.1406 | 20500 | 0.6074        | -               |
| 17.1615 | 20525 | 0.5221        | -               |
| 17.1824 | 20550 | 0.5144        | -               |
| 17.2033 | 20575 | 0.5419        | -               |
| 17.2242 | 20600 | 0.5716        | -               |
| 17.2451 | 20625 | 0.6052        | -               |
| 17.2661 | 20650 | 0.6153        | -               |
| 17.2870 | 20675 | 0.5803        | -               |
| 17.3079 | 20700 | 0.5157        | -               |
| 17.3288 | 20725 | 0.6344        | -               |
| 17.3497 | 20750 | 0.529         | -               |
| 17.3706 | 20775 | 0.495         | -               |
| 17.3915 | 20800 | 0.565         | -               |
| 17.4125 | 20825 | 0.566         | -               |
| 17.4334 | 20850 | 0.5206        | -               |
| 17.4543 | 20875 | 0.5201        | -               |
| 17.4752 | 20900 | 0.5462        | -               |
| 17.4961 | 20925 | 0.5535        | -               |
| 17.5170 | 20950 | 0.5673        | -               |
| 17.5380 | 20975 | 0.6024        | -               |
| 17.5589 | 21000 | 0.5351        | -               |
| 17.5798 | 21025 | 0.5469        | -               |
| 17.6007 | 21050 | 0.5683        | -               |
| 17.6216 | 21075 | 0.5636        | -               |
| 17.6425 | 21100 | 0.5921        | -               |
| 17.6635 | 21125 | 0.5186        | -               |
| 17.6844 | 21150 | 0.5723        | -               |
| 17.7053 | 21175 | 0.5483        | -               |
| 17.7262 | 21200 | 0.5363        | -               |
| 17.7471 | 21225 | 0.5329        | -               |
| 17.7680 | 21250 | 0.6122        | -               |
| 17.7890 | 21275 | 0.6264        | -               |
| 17.8099 | 21300 | 0.5063        | -               |
| 17.8308 | 21325 | 0.5015        | -               |
| 17.8517 | 21350 | 0.5497        | -               |
| 17.8726 | 21375 | 0.5035        | -               |
| 17.8935 | 21400 | 0.5328        | -               |
| 17.9145 | 21425 | 0.6012        | -               |
| 17.9354 | 21450 | 0.5722        | -               |
| 17.9563 | 21475 | 0.5171        | -               |
| 17.9772 | 21500 | 0.5223        | -               |
| 17.9981 | 21525 | 0.533         | -               |
| 18.0    | 21528 | -             | 0.4656          |
| 18.0184 | 21550 | 0.4626        | -               |
| 18.0393 | 21575 | 0.5182        | -               |
| 18.0602 | 21600 | 0.4869        | -               |
| 18.0812 | 21625 | 0.567         | -               |
| 18.1021 | 21650 | 0.4912        | -               |
| 18.1230 | 21675 | 0.51          | -               |
| 18.1439 | 21700 | 0.5789        | -               |
| 18.1648 | 21725 | 0.5465        | -               |
| 18.1857 | 21750 | 0.4983        | -               |
| 18.2067 | 21775 | 0.4962        | -               |
| 18.2276 | 21800 | 0.5357        | -               |
| 18.2485 | 21825 | 0.4724        | -               |
| 18.2694 | 21850 | 0.5552        | -               |
| 18.2903 | 21875 | 0.5224        | -               |
| 18.3112 | 21900 | 0.5361        | -               |
| 18.3321 | 21925 | 0.5436        | -               |
| 18.3531 | 21950 | 0.5906        | -               |
| 18.3740 | 21975 | 0.5707        | -               |
| 18.3949 | 22000 | 0.5731        | -               |
| 18.4158 | 22025 | 0.4826        | -               |
| 18.4367 | 22050 | 0.5351        | -               |
| 18.4576 | 22075 | 0.5536        | -               |
| 18.4786 | 22100 | 0.5785        | -               |
| 18.4995 | 22125 | 0.4802        | -               |
| 18.5204 | 22150 | 0.5677        | -               |
| 18.5413 | 22175 | 0.4921        | -               |
| 18.5622 | 22200 | 0.45          | -               |
| 18.5831 | 22225 | 0.5217        | -               |
| 18.6041 | 22250 | 0.5597        | -               |
| 18.6250 | 22275 | 0.5281        | -               |
| 18.6459 | 22300 | 0.505         | -               |
| 18.6668 | 22325 | 0.4846        | -               |
| 18.6877 | 22350 | 0.4788        | -               |
| 18.7086 | 22375 | 0.5556        | -               |
| 18.7296 | 22400 | 0.4912        | -               |
| 18.7505 | 22425 | 0.5394        | -               |
| 18.7714 | 22450 | 0.5671        | -               |
| 18.7923 | 22475 | 0.5359        | -               |
| 18.8132 | 22500 | 0.5352        | -               |
| 18.8341 | 22525 | 0.4932        | -               |
| 18.8551 | 22550 | 0.4821        | -               |
| 18.8760 | 22575 | 0.5511        | -               |
| 18.8969 | 22600 | 0.5399        | -               |
| 18.9178 | 22625 | 0.4942        | -               |
| 18.9387 | 22650 | 0.6056        | -               |
| 18.9596 | 22675 | 0.5248        | -               |
| 18.9805 | 22700 | 0.5419        | -               |
| 19.0    | 22724 | -             | 0.4563          |
| 19.0008 | 22725 | 0.5153        | -               |
| 19.0218 | 22750 | 0.5248        | -               |
| 19.0427 | 22775 | 0.5804        | -               |
| 19.0636 | 22800 | 0.4666        | -               |
| 19.0845 | 22825 | 0.5226        | -               |
| 19.1054 | 22850 | 0.4777        | -               |
| 19.1263 | 22875 | 0.4965        | -               |
| 19.1472 | 22900 | 0.5092        | -               |
| 19.1682 | 22925 | 0.4561        | -               |
| 19.1891 | 22950 | 0.5117        | -               |
| 19.2100 | 22975 | 0.5064        | -               |
| 19.2309 | 23000 | 0.5275        | -               |
| 19.2518 | 23025 | 0.5787        | -               |
| 19.2727 | 23050 | 0.4781        | -               |
| 19.2937 | 23075 | 0.5018        | -               |
| 19.3146 | 23100 | 0.4826        | -               |
| 19.3355 | 23125 | 0.5979        | -               |
| 19.3564 | 23150 | 0.5208        | -               |
| 19.3773 | 23175 | 0.5574        | -               |
| 19.3982 | 23200 | 0.4696        | -               |
| 19.4192 | 23225 | 0.4809        | -               |
| 19.4401 | 23250 | 0.5068        | -               |
| 19.4610 | 23275 | 0.5357        | -               |
| 19.4819 | 23300 | 0.5165        | -               |
| 19.5028 | 23325 | 0.5535        | -               |
| 19.5237 | 23350 | 0.5568        | -               |
| 19.5447 | 23375 | 0.531         | -               |
| 19.5656 | 23400 | 0.4597        | -               |
| 19.5865 | 23425 | 0.5423        | -               |
| 19.6074 | 23450 | 0.5408        | -               |
| 19.6283 | 23475 | 0.532         | -               |
| 19.6492 | 23500 | 0.5067        | -               |
| 19.6702 | 23525 | 0.5743        | -               |
| 19.6911 | 23550 | 0.5698        | -               |
| 19.7120 | 23575 | 0.4768        | -               |
| 19.7329 | 23600 | 0.4896        | -               |
| 19.7538 | 23625 | 0.5015        | -               |
| 19.7747 | 23650 | 0.4974        | -               |
| 19.7956 | 23675 | 0.5141        | -               |
| 19.8166 | 23700 | 0.4839        | -               |
| 19.8375 | 23725 | 0.4741        | -               |
| 19.8584 | 23750 | 0.5635        | -               |
| 19.8793 | 23775 | 0.5305        | -               |
| 19.9002 | 23800 | 0.5663        | -               |
| 19.9211 | 23825 | 0.5065        | -               |
| 19.9421 | 23850 | 0.4644        | -               |
| 19.9630 | 23875 | 0.4833        | -               |
| 19.9839 | 23900 | 0.4515        | -               |

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