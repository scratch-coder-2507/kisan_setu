# **20-CodePredators**

## Work Updates :

### PPT Link: [https://docs.google.com/presentation/d/1KxU9dkPCCIHPwoqqw_j9-pIPZ5mcDaBGgVrIaDl25Ag/edit?usp=sharing](https://docs.google.com/presentation/d/1KxU9dkPCCIHPwoqqw_j9-pIPZ5mcDaBGgVrIaDl25Ag/edit?usp=sharing)

### Yield Prediciton Model
![yield-prediciton-model Output - 1](https://res.cloudinary.com/devdemo/image/upload/v1666475315/VCET%20Hack/1_ienk8y.jpg)

![yield-prediciton-model Output - 2](https://res.cloudinary.com/devdemo/image/upload/v1666475309/VCET%20Hack/2_ytrlwc.jpg)

<img width='250' height='200'  src="https://camo.githubusercontent.com/5ceadc94fd40688144b193fd8ece2b805d79ca9b/68747470733a2f2f6c61726176656c2e636f6d2f6173736574732f696d672f636f6d706f6e656e74732f6c6f676f2d6c61726176656c2e737667">
</p>
<p>
<img width='250' height='100' src="http://scikit-learn.org/stable/_static/scikit-learn-logo-small.png">
</p>

## Running project/Contributing
You are welcome to contribute to this project,
Thank you for considering contributing for the greater good for farmers! 

**steps**
- you must to have [composer](composer.io) installed
- fork or clone project
- run `npm install` - for installing JS dependencies
- run `composer install` - for Installing Laravel dependencies
- run `cp .env.example .env` - copy `.env.example` and create `.env` file
- run `php artisan key:generate`
- run `php artisan serve` , this will serve project at `localhost:8000`
- that's it happy coding..

### Diseases Prediciton Model
![diseases-prediciton-model Output - 1](https://res.cloudinary.com/devdemo/image/upload/v1666475308/VCET%20Hack/7_uw6ggb.jpg)

![diseases-prediciton-model Output - 2](https://res.cloudinary.com/devdemo/image/upload/v1666475309/VCET%20Hack/6_vubbc3.jpg)

The API returns the json response in the following format:

```json
{
    "image_1": {
        "description": "description_1",
        "prediction": "prediction_1",
        "source": "source_link_1",
        "symptoms": "symptoms_1"
    },
    "image_2": {
        "description": "description_2",
        "prediction": "prediction_2",
        "source": "source_link_2",
        "symptoms": "symptoms_2"
    }
}
```

</details>

<details>
<summary>List of Crops and Diseases supported</summary>

- Apple
  - Apple Scab
  - Black Rot
  - Cedar Rust
  - Healthy
- Blueberry
  - Healthy
- Cherry
  - Powdery Mildew
  - Healthy
- Corn (Maize)
  - Grey Leaf Spot
  - Common Rust of Maize
  - Northern Leaf Blight
  - Healthy
- Grape
  - Black Rot
  - Black Measles (Esca)
  - Leaf Blight (Isariopsis Leaf Spot)
  - healthy
- Orange
  - Huanglongbing (Citrus Greening)
- Peach
  - Bacterial spot
  - healthy
- Bell Pepper
  - Bacterial Spot
  - Healthy
- Potato
  - Early Blight
  - Late Blight
  - Healthy
- Raspberry
  - Healthy
- Rice
  - Brown Spot
  - Hispa
  - Leaf Blast
  - Healthy
- Soybean
  - Healthy
- Squash
  - Powdery Mildew
- Strawberry
  - Leaf Scorch
  - Healthy
- Tomato
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites (Two-spotted Spider Mite)
  - Target Spot
  - Yellow Leaf Curl Virus
  - Mosaic Virus
  - Healthy

</details>

### Crop Prediciton Model
![Crop-prediciton-model Output - 1](https://res.cloudinary.com/devdemo/image/upload/v1666475307/VCET%20Hack/3_hasgfn.jpg)

![Crop-prediciton-model Output - 2](https://res.cloudinary.com/devdemo/image/upload/v1666475312/VCET%20Hack/5_g6vw88.jpg)

### Crop Quality
![Crop-quality-model Output - 1](https://res.cloudinary.com/devdemo/image/upload/v1666475309/VCET%20Hack/8_bwa2ek.jpg)

