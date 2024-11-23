<!DOCTYPE html>
<html lang="en">
<head>
    <title>CPP Home Page</title>
    <!-- Add Bootstrap CSS -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"> <!-- Font Awesome -->
    <style>
        /* Add your custom styles here */
        .button-container {
            position: relative; /* Position parent relatively */
        }
        .apply-now-btn {
            position: absolute;
            top: 245px; /* Moves the button down by 100 pixels */
            left: 50%; /* Optional: center horizontally if needed */
            transform: translateX(-50%); /* Optional: center horizontally */
        }
        body {
            background: #f4f4f9; /* Light gray background for the entire page */
        }
        .marquee-container {
            background-color: transparent;
            padding: 10px;
            overflow: hidden;
            position: relative;
        }
        .marquee-content {
            display: inline-block;
            white-space: nowrap;
            animation: marquee 10s linear infinite;
            color: #007bff;
        }
        @keyframes marquee {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        .flash-msg {
            margin-top: 20px;
        }
        .flash-msg .alert {
            margin-bottom: 0;
        }
        .container {
            margin-top: 60px;
        }
        .carousel-item img {
            height:100vh;
            width:300%; /* Set height to 50% of the viewport height */
            object-fit: cover; /* Ensure images cover the area without stretching */
        }
        .footer {
            background-color: #343a40;
            color: white;
            padding: 20px 0;
            text-align: center;
            position: relative;
            width: 100%;
            bottom: 0;
        }
        .footer a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .contact-location-btns {
            position: absolute;
            top: 100px; /* Adjusted top position to move buttons down slightly */
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .contact-location-btns a {
            display: block;
            width: 50px;
            height: 50px;
            background-color: white;
            border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            text-align: center;
            line-height: 50px;
            font-size: 24px;
            color: white;
            text-decoration: none;
        }
        .contact-location-btns .contact-btn {
            background-color: #007bff;
        }
        .contact-location-btns .location-btn {
            background-color: #dc3545;
        }
        .contact-location-btns .doctor-btn {
            background-color: #28a745;
        }
        .contact-location-btns .contact-btn:hover,
        .contact-location-btns .location-btn:hover,
        .contact-location-btns .doctor-btn:hover {
            opacity: 0.8;
        }
        .social-media-btns {
            position: absolute;
            top: 280px; /* Adjusted top position to move buttons down slightly */
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .social-media-btns a {
            display: block;
            width: 50px;
            height: 50px;
            background-color: #ffffff;
            border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            text-align: center;
            line-height: 50px;
            font-size: 24px;
            color: #333;
            text-decoration: none;
        }
        .social-media-btns .twitter-btn {
            background-color: #1da1f2;
            color: white;
        }
        .social-media-btns .instagram-btn {
            background-color: #e4405f;
            color: white;
        }
        .social-media-btns a:hover {
            opacity: 0.8;
        }
        .about-us {
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
        }
        .event-section {
            margin-top: 40px;
        }
        .event-section img {
            width: 100%;
            height: auto;
            cursor: pointer;
        }
        .login-section {
            margin-top: 40px;
        }
        .login-section img {
            width: 100px;
            height: auto;
        }
    </style>
</head>
<body>

    <?php include("include/header.php"); ?>

    <!-- Social Media Buttons -->
    <div class="social-media-btns">
        <a href="https://facebook.com" target="_blank" class="facebook-btn" title="Facebook">
            <i class="fab fa-facebook-f"></i>
        </a>
        <a href="https://twitter.com" target="_blank" class="twitter-btn" title="Twitter">
            <i class="fab fa-twitter"></i>
        </a>
        <a href="https://instagram.com" target="_blank" class="instagram-btn" title="Instagram">
            <i class="fab fa-instagram"></i>
        </a>
    </div>

    <!-- Running Flash Message -->
    <div class="marquee-container">
        <div class="marquee-content">
            You Can Predict Crop Yield And Price Prediction Here!!!
        </div>
    </div>

    <div class="container">
        <!-- Bootstrap Carousel -->
        <div id="carouselExampleCaptions" class="carousel slide" data-ride="carousel" data-interval="2000">
            <ol class="carousel-indicators">
                <li data-target="#carouselExampleCaptions" data-slide-to="0" class="active"></li>
                <li data-target="#carouselExampleCaptions" data-slide-to="1"></li>
                <li data-target="#carouselExampleCaptions" data-slide-to="2"></li>
            </ol>
            <div class="carousel-inner">
                <div class="carousel-item active">
                    <img src="https://www.agrivi.com/wp-content/uploads/2016/03/sl1-1024x517.jpg" class="d-block w-100" alt="First slide">
                    <div class="carousel-caption d-none d-md-block"></div>
                </div>
                <div class="carousel-item">
                    <img src="https://www.ifm.org/wp-content/uploads/Farmer_with_vegetables-scaled.jpg" class="d-block w-100" alt="Second slide">
                    <div class="carousel-caption d-none d-md-block"></div>
                </div>
                <div class="carousel-item">
                    <img src="https://www.investopedia.com/thmb/af2_szAGyG2cxHvdXISlmhi2Rg4=/3100x2067/filters:fill(auto,1)/harvest-164458970-505d26b04f134939a829746343346ec8.jpg" class="d-block w-100" alt="Third slide">
                    <div class="carousel-caption d-none d-md-block"></div>
                </div>
            </div>
            <a class="carousel-control-prev" href="#carouselExampleCaptions" role="button" data-slide="prev">
                <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                <span class="sr-only">Previous</span>
            </a>
            <a class="carousel-control-next" href="#carouselExampleCaptions" role="button" data-slide="next">
                <span class="carousel-control-next-icon" aria-hidden="true"></span>
                <span class="sr-only">Next</span>
            </a>
        </div>
        <br>
        <br>
        <br>
        <br>
        <br>
        <br>


        <!-- Event Section -->
        <div class="event-section">
            <h2 class="text-center">Crop Growth Guide</h2><br><br>
            <div class="row">
                <div class="col-md-3">
                    <img src="https://d30ifv2c9xs6en.cloudfront.net/public/cropGuidePdf/19/m_en-ph_icgCropGuidePdf_19_2.jpg" alt="Event 4" class="img-fluid" data-toggle="modal" data-target="#eventModal1">
                </div>
                <div class="col-md-3">
                    <img src="https://d30ifv2c9xs6en.cloudfront.net/public/cropGuidePdf/28/m_en-tz_icgCropGuidePdf_28_3.jpg" alt="Event 2" class="img-fluid" data-toggle="modal" data-target="#eventModal2">
                </div>
                <div class="col-md-3">
                    <img src="https://d30ifv2c9xs6en.cloudfront.net/public/cropGuidePdf/42/m_en_icgCropGuidePdf_42_2.jpg" alt="Event 3" class="img-fluid" data-toggle="modal" data-target="#eventModal3">
                </div>
                  <div class="col-md-3">
                    <img src="https://d30ifv2c9xs6en.cloudfront.net/public/cropGuidePdf/42/m_en_icgCropGuidePdf_42_2.jpg" alt="Event 4" class="img-fluid" data-toggle="modal" data-target="#eventModal4">
                </div>
            </div>
        </div>
        <br>
        <br>
    </div>
    <br>
    <br>

    <!-- Modal for Event 1 -->
    <div class="modal fade" id="eventModal1" tabindex="-1" role="dialog" aria-labelledby="eventModalLabel1" aria-hidden="true">
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="eventModalLabel1">Event 1</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <img src="event4.jpg" class="img-fluid" alt="Event 4">
                </div>
            </div>
        </div>
    </div>

    <!-- Modal for Event 2 -->
    <div class="modal fade" id="eventModal2" tabindex="-1" role="dialog" aria-labelledby="eventModalLabel2" aria-hidden="true">
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="eventModalLabel2">Event 2</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <img src="img/event2.jpg" class="img-fluid" alt="Event 2">
                </div>
            </div>
        </div>
    </div>

    <!-- Modal for Event 3 -->
    <div class="modal fade" id="eventModal3" tabindex="-1" role="dialog" aria-labelledby="eventModalLabel3" aria-hidden="true">
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="eventModalLabel3">Event 3</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <img src="img/event3.jpg" class="img-fluid" alt="Event 3">
                </div>
            </div>
        </div>
    </div>

    <!-- Modal for Event 4 -->
    <div class="modal fade" id="eventModal4" tabindex="-1" role="dialog" aria-labelledby="eventModalLabel4" aria-hidden="true">
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="eventModalLabel4">Event 4</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <img src="img/event4.jpg" class="img-fluid" alt="Event 4">
                </div>
            </div>
        </div>
    </div>

    <!-- About Us, Our Mission, and What Sets Us Apart Sections -->
    <div class="row">
        <!-- About Us Section -->
        <div class="col-md-4 text-center">
            <h2>About Us</h2>
            <p>Welcome to Crop Price Estimation, your trusted partner in crop price estimation and agricultural intelligence. At [Your Company Name], we understand the challenges farmers, traders, and stakeholders face in predicting crop prices in an ever-evolving agricultural landscape. Our mission is to empower the agricultural community with accurate, data-driven insights to make informed decisions, reduce risks, and maximize profitability.</p>
            <p>We are a team of passionate data scientists, agricultural experts, and technology enthusiasts dedicated to revolutionizing the agricultural market. Leveraging cutting-edge machine learning algorithms and advanced data analytics, we provide reliable crop price predictions, helping you navigate the complexities of the market with ease.</p>
        </div>

        <!-- Our Mission Section -->
        <div class="col-md-4 text-center">
            <h2>Our Mission</h2>
            <p>Our vision is to bridge the gap between farmers and market intelligence, making agriculture more transparent, predictable, and profitable. We aim to support sustainable farming practices and enhance food security by providing actionable insights and forecasts based on real-time data analysis.</p>
        </div>

        <!-- What Sets Us Apart Section -->
        <div class="col-md-4">
            <h2>What Sets Us Apart</h2>
            <ul>
               <li><strong>State-of-the-Art Agricultural Analytics:</strong> We utilize the latest machine learning models and data analytics tools to provide precise crop price forecasts, ensuring you stay ahead in the market.</li>
        
        <li><strong>Expert Team of Agricultural Data Scientists:</strong> Our dedicated team of experts combines knowledge in agriculture, data science, and technology to offer accurate and insightful predictions.</li>
        
        <li><strong>Comprehensive and Personalized Market Insights:</strong> We deliver tailored crop price predictions and detailed market analysis, helping farmers and traders make informed decisions for better profitability.</li>
        
        <li><strong>Commitment to Innovation and Accuracy:</strong> We continuously refine our predictive models, integrating new data sources and improving accuracy to adapt to the dynamic agricultural market.</li>
        
        <li><strong>Community Engagement and Farmer Education:</strong> We believe in empowering the agricultural community by providing training, resources, and educational programs on market trends and data-driven decision-making.</li>
            </ul>
        </div>
    </div>
    <br>
    <br>

    <!-- Footer with Address Bar and Clickable Links -->
    <footer class="footer">
        <p>No:378 Crop Price Estimation, Chennai, Tamil Nadu, India 602024 (Ph:01234567)</p>
        <p> 
           
            <a href="info.php">Founder</a>
        </p>
        <!-- <p>&copy; 2024 VH Groups & Hospitals. All rights reserved.</p> -->
    </footer>

    <!-- Add Bootstrap JS and

 dependencies -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>