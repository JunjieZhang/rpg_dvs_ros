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
    MILLISECONDS, NUM_EVENTS
  } frame_size_unit_;

  double frame_size_;

  ImageTracking image_tracking_;
};

} // namespace

#endif // DVS_RENDERER_H_
