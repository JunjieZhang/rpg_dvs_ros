// This file is part of DVS-ROS - the RPG DVS ROS Package
//
// DVS-ROS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DVS-ROS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with DVS-ROS.  If not, see <http://www.gnu.org/licenses/>.

#include "dvs_renderer/renderer.h"
#include <std_msgs/Float32.h>
#include <thread>
#include <opencv2/calib3d/calib3d.hpp>

namespace dvs_renderer {

Renderer::Renderer(ros::NodeHandle & nh, ros::NodeHandle nh_private) : nh_(nh),
    image_tracking_(nh)
{
  // setup subscribers and publishers
  event_sub_ = nh_.subscribe("events", 1, &Renderer::eventsCallback, this);
  camera_info_sub_ = nh_.subscribe("camera_info", 1, &Renderer::cameraInfoCallback, this);

  image_transport::ImageTransport it_(nh_);
  image_sub_ = it_.subscribe("image", 1, &Renderer::imageCallback, this);
  image_pub_ = it_.advertise("dvs_rendering", 1);
  undistorted_image_pub_ = it_.advertise("dvs_undistorted", 1);
  image_difference_pub_ = it_.advertise("image_difference", 1);

  // Dynamic reconfigure
  dynamic_reconfigure_callback_ = boost::bind(&Renderer::changeParameterscallback, this, _1, _2);
  server_.reset(new dynamic_reconfigure::Server<dvs_renderer::DVS_RendererConfig>(nh_private));
  server_->setCallback(dynamic_reconfigure_callback_);

  event_stats_[0].events_mean_[0] = nh_.advertise<std_msgs::Float32>("events_on_mean_1", 1);
  event_stats_[0].events_mean_[1] = nh_.advertise<std_msgs::Float32>("events_off_mean_1", 1);
  event_stats_[1].events_mean_[0] = nh_.advertise<std_msgs::Float32>("events_on_mean_5", 1);
  event_stats_[1].events_mean_[1] = nh_.advertise<std_msgs::Float32>("events_off_mean_5", 1);

  frame_rate_hz_ = 25;
  changed_frame_rate_ = true;
  synchronize_on_frames_ = false;

  reset();

  std::thread renderThread(&Renderer::renderFrameLoop, this);
  renderThread.detach();
}

Renderer::~Renderer()
{
  image_pub_.shutdown();
  undistorted_image_pub_.shutdown();
  image_difference_pub_.shutdown();
}

void Renderer::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
  got_camera_info_ = true;
  distortion_model_ = msg->distortion_model;
  cv::Size sensor_size(msg->width, msg->height);

  camera_matrix_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      camera_matrix_.at<double>(cv::Point(i, j)) = msg->K[i+j*3];

  dist_coeffs_ = cv::Mat(msg->D.size(), 1, CV_64F);
  for (int i = 0; i < msg->D.size(); i++)
    dist_coeffs_.at<double>(i) = msg->D[i];

  if(distortion_model_ == "equidistant")
  {
    cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                         (cv::Mat) cv::Matx33d::eye(), camera_matrix_,
                                         sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
  }
  else
  {
    cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                (cv::Mat) cv::Matx33d::eye(), camera_matrix_,
                                sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
  }
}

void Renderer::imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
  if(sensor_size_.width <= 0)
  {
    init(msg->width, msg->height);
  }

  image_tracking_.imageCallback(msg);

  ROS_DEBUG("Image buffer size: %d", images_.size());
  cv_bridge::CvImagePtr cv_ptr;

  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    images_.insert(StampedImage(msg->header.stamp, cv_ptr->image));
    clearImageBuffer();
  }

  if(synchronize_on_frames_)
  {
    renderAndPublishImageAtTime(msg->header.stamp);
  }
}

void Renderer::changeParameterscallback(dvs_renderer::DVS_RendererConfig &config, uint32_t level) {
  std::lock_guard<std::mutex> lock(data_mutex_);

  ROS_DEBUG("Received parameter change callback.\n display_method=%d\n frame_size_unit=%d\n frame_size=%d\n synchronize_on_frames=%d\n use_only_events_between_frames=%d\n frame_rate=%d\n median_blur=%d\n",
          config.use_color, config.rendering_method, config.frame_size, config.synchronize_on_frames, config.use_events_between_frames, config.frame_rate, config.median_blur);
  display_method_ = (config.use_color) ? RED_BLUE : GRAYSCALE;
  frame_size_unit_ = (FrameSizeUnit) config.rendering_method;
  frame_size_ = (double) config.frame_size;
  synchronize_on_frames_ = config.synchronize_on_frames;
  use_only_events_between_frames_ = config.use_events_between_frames;

  if(use_only_events_between_frames_ && !synchronize_on_frames_)
  {
    ROS_WARN("Cannot use only the events between frames when not in synchronize_on_frames mode.");
    use_only_events_between_frames_ = false;
  }

  median_blur_kernel_size_ = config.median_blur;

  if(config.frame_rate != frame_rate_hz_)
  {
    changed_frame_rate_ = true;
    frame_rate_hz_ = config.frame_rate;
  }
}

void Renderer::init(int width, int height)
{
  sensor_size_ = cv::Size(width, height);
  ROS_INFO("Initialized with size: (%d x %d)", sensor_size_.width, sensor_size_.height);

  static constexpr int max_num_events_in_history_per_pixel = 20;
  events_history_.reset(new EventHistoryMap(width, height, max_num_events_in_history_per_pixel));
}

void Renderer::renderFrameLoop()
{
  ros::Rate r(frame_rate_hz_);

  while (ros::ok())
  {
    if(changed_frame_rate_)
    {
      ROS_INFO("Changing framerate to %d Hz", frame_rate_hz_);
      r = ros::Rate(frame_rate_hz_);
      changed_frame_rate_ = false;
    }

    if(!synchronize_on_frames_)
    {
      renderAndPublishImageAtTime(events_.back().ts);
    }

    r.sleep();
  }
}

void Renderer::renderAndPublishImageAtTime(const ros::Time& frame_end_stamp)
{
  if(sensor_size_.width <= 0 || sensor_size_.height <= 0 || events_.size() < 2)
    return;

  cv::Mat event_img, event_img_color;

  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if(frame_size_unit_ == NUM_EVENTS || frame_size_unit_ == MICROSECONDS)
    {
      EventBuffer::iterator it_frame_start;
      EventBuffer::iterator it_frame_end = firstEventOlderThan(frame_end_stamp);

      if(images_.size() >= 2 && use_only_events_between_frames_)
      {
        auto it = images_.rbegin();
        it--;
        it_frame_start = firstEventOlderThan(it->first);
        ROS_INFO("Frame start -> frame end: %f -> %f", it_frame_start->ts.toSec(), it_frame_end->ts.toSec());
      }
      else if(frame_size_unit_ == MICROSECONDS)
      {
        const double frame_duration_s = frame_size_ / 1000000.0;
        it_frame_start = firstEventOlderThan(frame_end_stamp - ros::Duration(frame_duration_s));
      }
      else
      {
        const size_t num_events = static_cast<size_t>(frame_size_);
        if(std::distance(events_.begin(), it_frame_end) < num_events)
        {
          it_frame_start = events_.begin();
        }
        else
        {
          it_frame_start = it_frame_end - num_events;
        }
      }

      if(display_method_ == RED_BLUE)
      {
        ROS_DEBUG("Query stamp: %f", events_.back().ts.toSec());
        if(images_.empty())
        {
          event_img = cv::Mat::zeros(sensor_size_, CV_8U);
        }
        else
        {
          static constexpr double max_img_delay_s = 0.5;
          const ros::Time last_event_stamp = events_.back().ts;
          const ros::Time last_img_stamp = images_.rbegin()->first;
          ROS_DEBUG("Image stamp: %f", last_img_stamp);
          if(synchronize_on_frames_ || (std::fabs(last_img_stamp.toSec() - last_event_stamp.toSec()) <= max_img_delay_s))
          {
            event_img = images_.rbegin()->second;
          }
          else
          {
            event_img = cv::Mat::zeros(sensor_size_, CV_8U);
          }
        }

        event_img.convertTo(event_img_color, CV_8UC3, 1.0, 0.0);
        cv::cvtColor(event_img_color, event_img_color, cv::COLOR_GRAY2BGR);
        drawEventsColor(it_frame_start, it_frame_end, &event_img_color, true);
      }
      else
      {
        event_img = cv::Mat::zeros(sensor_size_, CV_32F);
        drawEventsGrayscale(it_frame_start, it_frame_end, &event_img);

        static constexpr double bmax = 5.0;
        event_img = 255.0 * (event_img + bmax) / (2.0 * bmax);
        event_img.convertTo(event_img, CV_8U, 1.0, 0.0);
      }
    }
    else
    {
      if(use_only_events_between_frames_)
      {
        ROS_WARN("use_only_events_between_frames mode is incompatible with the dI/dt mode. Will ignore that option.");
      }
      const bool use_polarity = true;
      const double decay_s = frame_size_ / 1000000.0;
      event_img = cv::Mat::zeros(sensor_size_, CV_64F);
      for(int y=0; y<sensor_size_.height; ++y)
      {
        for(int x=0; x<sensor_size_.width; ++x)
        {
          dvs_msgs::Event first_event_at_xy_before_frame_end;
          if(events_history_->first_event_at_xy_older_than_t(x, y, frame_end_stamp, &first_event_at_xy_before_frame_end))
          {
            const ros::Time& last_stamp_at_xy = first_event_at_xy_before_frame_end.ts;
            double dIdt = 0.0;
            if(last_stamp_at_xy.toSec() > 0)
            {
              const double dt_s = (frame_end_stamp - last_stamp_at_xy).toSec();
              const double pol = (first_event_at_xy_before_frame_end.polarity) ? 1.0 : -1.0;
              dIdt = std::exp(-dt_s / decay_s);
              if(use_polarity)
              {
                dIdt *= pol;
              }
              event_img.at<double>(y,x) = dIdt;
            }
          }
        }
      }
      if(use_polarity)
      {
        event_img = 255.0 * (event_img + 1.0) / 2.0;
      }
      else
      {
        event_img = 255.0 * event_img;
      }
      event_img.convertTo(event_img, CV_8U);
    }
  }

  // Publish event image
  static cv_bridge::CvImage cv_image;

  if(frame_size_unit_ != PARTIAL_DI_DT && display_method_ == RED_BLUE)
  {
    cv_image.encoding = "bgr8";
    cv_image.image = event_img_color;
  }
  else
  {
    if(median_blur_kernel_size_ > 0)
    {
      cv::medianBlur(event_img, event_img, 2 * median_blur_kernel_size_ + 1);
    }

    cv_image.encoding = "mono8";
    cv_image.image = event_img;
  }
  cv_image.header.stamp = frame_end_stamp;
  image_pub_.publish(cv_image.toImageMsg());

  if (got_camera_info_ && undistorted_image_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImage cv_image2;
    cv_image2.encoding = cv_image.encoding;
    cv_image2.header.stamp = frame_end_stamp;
    cv::remap(cv_image.image, cv_image2.image, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
    undistorted_image_pub_.publish(cv_image2.toImageMsg());
  }

  // Publish the difference between the last two images
  if(images_.size() >= 2 && image_difference_pub_.getNumSubscribers() > 0)
  {
    cv::Mat last_image, second_to_last_image;
    static constexpr double eps = 0.001;
    getLastTwoImages(&last_image, &second_to_last_image);

    last_image.convertTo(last_image, CV_64F, 1.0/255.0, 0.0);
    cv::log(eps + last_image, last_image);

    second_to_last_image.convertTo(second_to_last_image, CV_64F, 1.0/255.0, 0.0);
    cv::log(eps + second_to_last_image, second_to_last_image);

    cv::Mat delta_logI = last_image - second_to_last_image;
    cv_bridge::CvImage cv_image3;
    cv_image3.encoding = "mono8";
    cv_image3.header.stamp = frame_end_stamp;

    const double max_delta_logI = -std::log(eps);
    delta_logI = 255.0 * (delta_logI + max_delta_logI) / (2.0 * max_delta_logI);
    delta_logI.convertTo(cv_image3.image, CV_8U);
    image_difference_pub_.publish(cv_image3.toImageMsg());
  }
}

void Renderer::drawEventsGrayscale(const EventBuffer::iterator& ev_first,
                                   const EventBuffer::iterator& ev_last,
                                   cv::Mat *out)
{
  CV_Assert(out->type() == CV_32F);

  // Draw Events
  auto draw = [] (float& p, const float val)
  {
    p = p+val;
  };

  for(EventBuffer::iterator e = ev_first; e != ev_last; ++e)
  {
    const float pol = (e->polarity) ? 1.f : -1.f;
    draw(out->at<float>(e->y,   e->x),   pol);
  }
}

void Renderer::drawEventsColor(const EventBuffer::iterator& ev_first,
                               const EventBuffer::iterator& ev_last,
                               cv::Mat *out,
                               bool use_polarity)
{
  CV_Assert(out->type() == CV_8UC3);

  // Draw Events
  auto draw = [] (cv::Vec3b& p, const cv::Vec3b val)
  {
    p = val;
  };

  for(EventBuffer::iterator e = ev_first; e != ev_last; ++e)
  {
    cv::Vec3b pol;

    if(use_polarity)
    {
      pol = ((e->polarity) ? cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 0, 255));
    }
    else
    {
      pol = cv::Vec3b(255, 0, 0);
    }
    draw(out->at<cv::Vec3b>(e->y,   e->x),   pol);
  }
}

void Renderer::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if(sensor_size_.width <= 0)
  {
    init(msg->width, msg->height);
  }

  // Check if the timestamps of the incoming events are inconsistent with the ones in the buffer.
  // This happens when a new sensor with a different time reference is plugged in, or
  // when a different rosbag is being played.
  // When that happens, reset the renderer.
  if(!msg->events.empty() && !events_.empty())
  {
    static constexpr double max_time_diff_before_reset_s = 0.5;
    const ros::Time stamp_first_event = msg->events[0].ts;
    const double dt = std::fabs(stamp_first_event.toSec() - events_.back().ts.toSec());
    if(dt >= max_time_diff_before_reset_s)
    {
      ROS_INFO("Inconsistent timestamps detected (new: %f, old: %f), resetting.", stamp_first_event.toSec(), events_.back().ts.toSec());
      reset();
      init(msg->width, msg->height);
    }
  }

  for(const dvs_msgs::Event& e : msg->events)
  {
    ++event_stats_[0].events_counter_[e.polarity];
    ++event_stats_[1].events_counter_[e.polarity];
    insertEventInSortedBuffer(e);
  }
  clearEventBuffer();

  publishStats();
  image_tracking_.eventsCallback(msg);
}

void Renderer::clearEventBuffer()
{
  static constexpr size_t event_history_size_ = 5000000;

  if (events_.size() > event_history_size_)
  {
    size_t remove_events = events_.size() - event_history_size_;

    events_.erase(events_.begin(),
                  events_.begin() + remove_events);
  }
}

void Renderer::clearImageBuffer()
{
  static constexpr size_t image_history_size_ = 30;

  if (images_.size() > image_history_size_)
  {
    size_t remove_images = images_.size() - image_history_size_;
    auto erase_iter = images_.begin();
    std::advance(erase_iter, remove_images);
    images_.erase(images_.begin(),
                  erase_iter);
  }
}

void Renderer::publishStats()
{
  std_msgs::Float32 msg;
  ros::Time now = ros::Time::now();
  for (int i = 0; i < 2; ++i)
  {
    if (event_stats_[i].events_mean_lasttime_ + event_stats_[i].dt <= now.toSec()) {
      event_stats_[i].events_mean_lasttime_ = now.toSec();
      for (int k = 0; k < 2; ++k)
      {
        msg.data = (float)event_stats_[i].events_counter_[k] / event_stats_[i].dt;
        event_stats_[i].events_mean_[k].publish(msg);
        event_stats_[i].events_counter_[k] = 0;
      }
    }
  }
}

} // namespace
