import React from 'react'

export const CheckBox = props => {
    return (
      <ul>
       <input key={props.id} onClick={props.handleCheckChieldElement} type="checkbox" checked={props.isChecked} value={props.value} />&nbsp;&nbsp;{props.value}
      </ul>
    )
}

export default CheckBox