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

#ifndef DVS_RENDERER_H_
#define DVS_RENDERER_H_

#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "image_tracking.h"

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <deque>
#include <mutex>

#include <dynamic_reconfigure/server.h>
#include <dvs_renderer/DVS_RendererConfig.h>

namespace dvs_renderer
{

using EventBuffer = std::deque<dvs_msgs::Event>;
using StampedImage = std::pair<ros::Time, cv::Mat>;
using ImageBuffer = std::map<ros::Time, cv::Mat>;
using TimestampMap = std::vector<ros::Time>;
using PolarityMap = std::vector<uchar>;

// A data structure storing the last K events for every pixel
class EventHistoryMap {
public:

  EventHistoryMap(int width, int height, int K)
  {
    width_ = width;
    height_ = height;
    K_ = K;
    events_ = std::vector<EventBuffer>(width_ * height_, EventBuffer());
  }

  void insertEvent(const dvs_msgs::Event& e)
  {
    if(!contains(e.x, e.y))
    {
      ROS_WARN("Tried to insert event at (%d, %d) which is out of bounds of the sensor (%d, %d)", e.x, e.y, width_, height_);
      return;
    }
    else
    {
      EventBuffer& events_at_xy = get_events_at_xy(e.x, e.y);
      events_at_xy.push_back(e);
      if(events_at_xy.size() > K_)
      {
        events_at_xy.pop_front();
      }
    }
  }

  bool first_event_at_xy_older_than_t(const size_t x, const size_t y, const ros::Time& t, dvs_msgs::Event* ev)
  {
    if(!contains(x, y))
    {
      ROS_WARN("Tried to query event at (%d, %d) which is out of bounds of the sensor (%d, %d)", x, y, width_, height_);
      return false;
    }

    EventBuffer& events_at_xy = get_events_at_xy(x, y);
    if(events_at_xy.empty())
    {
      return false;
    }

    for(auto it = events_at_xy.rbegin(); it != events_at_xy.rend(); ++it)
    {
      const dvs_msgs::Event& e = *it;
      if(e.ts < t)
      {
        *ev = *it;
        return true;
      }
    }
    return false;
  }

  void clear()
  {
    events_.clear();
  }

private:

  bool contains(const size_t x, const size_t y)
  {
    return !(x < 0 || x >= width_ || y < 0 || y >= height_);
  }

  inline EventBuffer& get_events_at_xy(const size_t x, const size_t y)
  {
    return events_[x + width_ * y];
  }

  size_t width_;
  size_t height_;
  size_t K_;
  std::vector<EventBuffer> events_;
};

class Renderer {
public:
  Renderer(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~Renderer();

private:
  ros::NodeHandle nh_;

  void init(int width, int height);

  void renderFrameLoop();

  void drawEventsGrayscale(const EventBuffer::iterator& ev_first,
                           const EventBuffer::iterator& ev_last,
                           cv::Mat *out);

  void drawEventsColor(const EventBuffer::iterator& ev_first,
                       const EventBuffer::iterator& ev_last,
                       cv::Mat *out,
                       bool use_polarity);

  void renderAndPublishImageAtTime(const ros::Time& frame_end_stamp);

  void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);
  void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
  void changeParameterscallback(dvs_renderer::DVS_RendererConfig &config, uint32_t level);

  void reset()
  {
    got_camera_info_ = false;
    events_.clear();
    images_.clear();

    if(events_history_)
      events_history_->clear();

    sensor_size_ = cv::Size(0,0);

    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 2; ++k)
        event_stats_[i].events_counter_[k] = 0;
    event_stats_[0].dt = 1;
    event_stats_[0].events_mean_lasttime_ = 0;
    event_stats_[1].dt = 5;
    event_stats_[1].events_mean_lasttime_ = 0;
  }

  void publishStats();

  // Insert an event in the buffer while keeping the buffer sorted
  // This uses insertion sort as the events already come almost always sorted
  inline void insertEventInSortedBuffer(const dvs_msgs::Event& e)
  {
    events_.push_back(e);
    // insertion sort to keep the buffer sorted
    // in practice, the events come almost always sorted,
    // so the number of iterations of this loop is almost always 0
    int j = (events_.size() - 1) - 1; // second to last element
    while(j >= 0 && events_[j].ts > e.ts)
    {
      events_[j+1] = events_[j];
      j--;
    }
    events_[j+1] = e;

    const dvs_msgs::Event& last_event = events_.back();
    events_history_->insertEvent(last_event);
  }

  void clearEventBuffer();
  void clearImageBuffer();

  inline EventBuffer::iterator firstEventOlderThan(const ros::Time& stamp)
  {
    auto it = std::lower_bound(events_.begin(),
                               events_.end(),
                               stamp,
                               [](const dvs_msgs::Event &ev, const ros::Time& t) -> bool
    {
      return ev.ts < t;
    });
    if(it == events_.begin())
    {
      return it;
    }
    it--;
    return it;
  }

  bool got_camera_info_;
  cv::Mat camera_matrix_, dist_coeffs_;
  std::string distortion_model_;
  cv::Mat undistort_map1_, undistort_map2_;

  ros::Subscriber event_sub_;
  ros::Subscriber camera_info_sub_;

  image_transport::Publisher image_pub_;
  image_transport::Publisher undistorted_image_pub_;

  image_transport::Subscriber image_sub_;

  boost::shared_ptr<dynamic_reconfigure::Server<dvs_renderer::DVS_RendererConfig> > server_;
  dynamic_reconfigure::Server<dvs_renderer::DVS_RendererConfig>::CallbackType dynamic_reconfigure_callback_;

  cv::Size sensor_size_;

  size_t frame_rate_hz_;
  bool synchronize_on_frames_;
  bool changed_frame_rate_;
  int median_blur_kernel_size_;

  std::shared_ptr<EventHistoryMap> events_history_;
  EventBuffer events_;
  ImageBuffer images_;
  std::mutex data_mutex_;

  struct EventStats {
    ros::Publisher events_mean_[2]; /**< event stats output */
    int events_counter_[2]; /**< event counters for on/off events */
    double events_mean_lasttime_;
    double dt;
  };
  EventStats event_stats_[2]; /**< event statistics for 1 and 5 sec */

  enum DisplayMethod
  {
    GRAYSCALE, RED_BLUE
  } display_method_;

  enum FrameSizeUnit
  {
    MICROSECONDS, NUM_EVENTS, PARTIAL_DI_DT
  } frame_size_unit_;

  double frame_size_;

  ImageTracking image_tracking_;
};

} // namespace

#endif // DVS_RENDERER_H_
